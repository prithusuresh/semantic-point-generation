import glob
import os

import torch
import tqdm
import numpy as np
from torch.nn.utils import clip_grad_norm_
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
from sklearn.metrics import accuracy_score 
from eval_utils.eval_utils import load_data_to_gpu
z_min = None
z_max = None
z_size = None
def create_pillar_to_voxel(point_cloud_range, voxel_size):
    global z_min
    global z_max
    global z_size
    z_min = point_cloud_range[2]
    z_max = point_cloud_range[2+3]
    z_size = voxel_size[-1]
    

def lookup_voxel(point):
    assert z_min is not None
    assert z_max is not None
    assert z_size is not None
    eps = 1e-10
    z = point[:, 2]
    voxel_coord = (z - z_min - eps)//z_size 
    return voxel_coord.long()

def spg_add_confidence_to_voxel(batch):
    voxels, voxel_coords = batch["voxels"], batch["voxel_coords"]
    confidence = batch["output_prob"]
    voxels = torch.nn.functional.pad(voxels, pad = (0,1))
    i = 0
    for v, coords in zip(voxels, voxel_coords):
        b,z,y,x = coords[0].long(),coords[1].long(),coords[2].long(),coords[3].long()
        true_points = (v.sum(axis = -1) != 0)
        z_coords = lookup_voxel(v)
        pillar = confidence[b,:,y,x]
        voxels[i][true_points,[-1 for i in range(true_points.sum())]] = pillar[z_coords[true_points]]
        i+=1
    batch["voxels"] = voxels
    return
def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False, spg_model = None):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        if spg_model != None:
            load_data_to_gpu(batch)
            batch = spg_model(batch)
            spg_add_confidence_to_voxel(batch)
        model.train()
        optimizer.zero_grad()

        loss, tb_dict, disp_dict = model_func(model, batch)

        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter += 1
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})

        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False, spg_model = None):
    
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,
                spg_model = spg_model
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)

def calculate_weighted_focal_loss(criterion, pred, gt, weight_mask, threshold, hidden_voxels):
    loss = criterion(pred, gt)
    weighted_loss = (weight_mask*loss).sum()
    pred_detached = pred.detach()
    pred_detached = torch.sigmoid(pred_detached)
    pred_detached[pred_detached >= threshold] = 1
    pred_detached[pred_detached < threshold] = 0
    hidden = hidden_voxels.long()
    b,z,y,x = hidden[:,0], hidden[:,1], hidden[:,2], hidden[:,3]
    gt_cpu_flatten = gt[b,z,y,x].cpu().numpy()
    pred_detached_cpu = pred_detached[b,z,y,x].cpu().numpy()

    #hidden voxel accuracy
    report = accuracy_score(gt_cpu_flatten.flatten(), pred_detached_cpu.flatten())
    
    return weighted_loss, report
def prepare_ground_truth_and_weight_mask(train_sample, cfg):
    # 0 -> Unoccupied Background
    # 1 -> Occupied Background
    # 2 -> Unoccupied Foreground
    # 3 -> Occupied Foreground
    
    OCCUPIED_FLAG = 1
    FOREGROUND_FLAG = 2
    HIDDEN = 4
    mask = torch.zeros_like(train_sample["output_prob"]).long()
    occupied = train_sample["small_voxel_coords"].long()
    b, z, y, x = occupied[:,0], occupied[:,1], occupied[:,2], occupied[:,3]
    mask[b,z,y,x] += OCCUPIED_FLAG 
    
    all_voxel_coords_reshaped = train_sample["all_voxel_coords"].reshape((train_sample["batch_size"], -1, 4))
    all_voxels_in_foreground = points_in_boxes_gpu(train_sample["all_voxel_centers"], train_sample["gt_boxes"][:,:,:7])
    foreground = all_voxel_coords_reshaped[all_voxels_in_foreground != -1].long()
    b, z, y, x = foreground[:,0], foreground[:,1], foreground[:,2], foreground[:,3]
    mask[b,z,y,x] += FOREGROUND_FLAG 

    hidden = train_sample["hidden_voxel_coords"].long()
    b,z,y,x = hidden[:,0], hidden[:,1], hidden[:,2], hidden[:,3]

    ground_truth = (mask > 1).long()
    weight_mask = mask.float()
    
    
    NUM_HIDDEN = len(hidden)
    NUM_OCCUPIED_AND_EMPTY_BACKGROUND = (mask != 2).sum() - NUM_HIDDEN
    NUM_EMPTY = (mask == 2).sum()

    weight_mask[(mask != 2)] = 1/NUM_OCCUPIED_AND_EMPTY_BACKGROUND #weight is 1
    weight_mask[mask == 2] = cfg.alpha/NUM_EMPTY
    weight_mask[b,z,y,x] = cfg.beta/NUM_HIDDEN

    return ground_truth, weight_mask