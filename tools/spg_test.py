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
from pprint import pprint
from tensorboardX import SummaryWriter
import pandas as pd
import numpy as np

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.train_utils import checkpoint_state, save_checkpoint, train_model, prepare_ground_truth_and_weight_mask
from eval_utils import eval_utils 
from spg_model import SPG_CLASSIFICATION

from eval_utils.eval_utils import load_data_to_gpu

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--spg_ckpt', type=str, default=None, help='specify val')

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

    
    # -----------------------create dataloader & network & optimizer---------------------------
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=None,
        training=False,
    )
    model = SPG_CLASSIFICATION()
    model.cuda()
    if os.path.exists(args.spg_ckpt):
        print ("Loading model")
        d = torch.load(args.spg_ckpt, map_location = "cuda")
        model.load_state_dict(d["model_state"])
    model.eval()
    from kornia.losses import BinaryFocalLossWithLogits
    
    criterion = BinaryFocalLossWithLogits(alpha = 0.25, gamma = 2.0, reduction = "none")

    pbar = tqdm(enumerate(train_loader), total = len(train_loader))
    report_total = []
    gt_labels = None
    out_probs = None
    columns = [
        '0_precision', 
        '0_recall', 
        '0_f1-score',
        '0_support', 
        '1_precision', 
        '1_recall', 
        '1_f1-score',
        '1_support', 
        'accuracy',
        'macro_precision',
        'macro_recall', 
        'macro_f1-score',
        'macro_support', 
        'weighted_precision',
        'weighted_recall',
        'weighted_f1-score', 
        'weighted_support'
        ]
    
    with torch.no_grad():
        
        for i,x in pbar:
            load_data_to_gpu(x)
            out = model(x)
            ground_truth, weight_mask = prepare_ground_truth_and_weight_mask(train_sample=out, cfg = cfg.OPTIMIZATION)
            loss, acc, gt, probs = eval_utils.calculate_weighted_focal_loss(
                criterion, 
                out["output_prob"], 
                ground_truth, 
                weight_mask, 
                cfg.OPTIMIZATION.FOREGROUND_THRESHOLD,
                out["hidden_voxel_coords"])
            if len(acc) < len(columns):
                continue
            if gt_labels is None:
                gt_labels = gt
            else:
                gt_labels = np.concatenate([gt_labels, gt])
            if out_probs is None:
                out_probs = probs
            else:
                out_probs = np.concatenate([out_probs, probs])
            report_total.append(acc)
            pbar.set_description("loss: {:.4f}".format(loss))
            pbar.update()
    #eval_utils.plot_roc_curve(gt_labels, out_probs)
    #eval_utils.plot_pr_curve(gt_labels, out_probs)
    
    np.save("gt_labels.npy", gt_labels)
    np.save("pred.npy",out_probs)
    #report_total = np.asarray(report_total)
    #df = pd.DataFrame(report_total, columns=columns)
    #df.to_csv("val_set_results.csv")

if __name__ == '__main__':
    main()
