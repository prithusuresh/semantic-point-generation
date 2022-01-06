import argparse
import datetime
import glob
import os
from pathlib import Path
import pickle
from numpy.lib.npyio import save
from test import repeat_eval_ckpt
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import calculate_weighted_focal_loss, checkpoint_state, save_checkpoint, train_model, prepare_ground_truth_and_weight_mask
from spg_model import SPG_CLASSIFICATION

from eval_utils.eval_utils import load_data_to_gpu

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

   # tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # -----------------------create dataloader & network & optimizer---------------------------
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=None,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs
    )
    
    model = SPG_CLASSIFICATION()
    model.cuda()
    
    
    from kornia.losses import BinaryFocalLossWithLogits
    
    criterion = BinaryFocalLossWithLogits(alpha = 0.25, gamma = 2.0, reduction = "none")
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)
    
    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=-1, optim_cfg=cfg.OPTIMIZATION
    )
    accumulated_iter = 0
    model.train()
    
    for epoch in range(cfg.OPTIMIZATION.NUM_EPOCHS):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        pbar = tqdm(enumerate(train_loader), total = len(train_loader))
        running_loss = 0
        if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
            cur_scheduler = lr_warmup_scheduler
        else:
            cur_scheduler = lr_scheduler
        for i,x in pbar:
            lr_scheduler.step(accumulated_iter)
        
            optimizer.zero_grad()
        
            load_data_to_gpu(x)
            out = model(x)
            ground_truth, weight_mask = prepare_ground_truth_and_weight_mask(train_sample=out, cfg = cfg.OPTIMIZATION)
            loss, acc = calculate_weighted_focal_loss(
                criterion, 
                out["output_prob"], 
                ground_truth, 
                weight_mask, 
                cfg.OPTIMIZATION.FOREGROUND_THRESHOLD,
                out["hidden_voxel_coords"])
        
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.OPTIMIZATION.GRAD_NORM_CLIP)
            optimizer.step()
            running_loss += loss.item()
            pbar.set_description("loss: {:.4f}acc: {:.4f}".format(running_loss/(i + 1), acc))
            pbar.update()
            accumulated_iter += 1
        if epoch > 0 and epoch%cfg.OPTIMIZATION.log_every_n_epochs == 0:
            state_dict = checkpoint_state(model = model, optimizer = optimizer, epoch=epoch, it = accumulated_iter)
            fname = ckpt_dir / ('checkpoint_epoch_%d' % epoch)
            save_checkpoint(state_dict, fname)

        running_loss /= (i+1)
        logger.info("EPOCH [{}] loss: {:.4f}".format(epoch, running_loss))
    logger.info('**********************End evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()
