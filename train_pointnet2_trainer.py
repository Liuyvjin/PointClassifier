#!/usr/bin/env python
# -*- coding: utf-8 -*-
import shutil
import argparse
import os.path as osp
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

import data.data_utils as dutil
from data.data_utils import ModelNet40, pointnet_train_transforms, pointnet_test_transforms
from models.pointnet2 import Pointnet2, get_loss
from pointfield.model import PF_1layer, PF_Nlayer, PF_Model, PF_Tnet
from pointfield.train_utils import Trainer
from pointfield.losses import OnlineTripletLoss
from pointfield.loss_utils import HardestNegativeTripletSelector, pdist_pc, chamfer_pc, RandomNegativeTripletSelector


BASE_DIR = osp.dirname(osp.abspath(__file__))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--model',          type=str,   default='pointnet2',    help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--exp_name',       type=str,   default='pointnet2/tnet_random_nor',   help='expriment name')
    parser.add_argument('--log_dir',        type=str,   default='new_logs',     help='log directory')
    parser.add_argument('--batch_size',     type=int,   default=48,     help='batch size in training [default: 24]')
    parser.add_argument('--num_points',     type=int,   default=1024,   help='Point Number [default: 1024]')
    parser.add_argument('--num_epochs',     type=int,   default=200,    help='number of epoch in training [default: 200]')
    parser.add_argument('--num_workers',    type=int,   default=4,      help='Worker Number [default: 8]')
    parser.add_argument('--optimizer',      type=str,   default='SGD', help='optimizer for training [default: Adam]')
    parser.add_argument('--lr',             type=float, default=0.001,  help='learning rate in training [default: 0.001, 0.1 if using sgd]')
    parser.add_argument('--normal',         type=bool,  default=True,   help='Whether to use normal information [default: True]')
    parser.add_argument('--seed',           type=int,   default=1,      help='random seed [efault: 1]')
    parser.add_argument('--decay_rate',     type=float, default=1e-4,   help='decay rate [default: 1e-4]')
    # pointfield
    parser.add_argument('--use_pointfield', type=bool,   default=True,         metavar='N', help='Num of nearest neighbors to use')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    return args



def main():
    #--- Training settings
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")

    # --- DataLoader
    print('Load dataset...')
    train_transforms = transforms.Compose([ dutil.translate_pointcloud,
                                            dutil.shuffle_pointcloud])

    # train_transforms = pointnet_train_transforms
    # test_transforms = pointnet_test_transforms
    train_loader = DataLoader(  ModelNet40(normal_channel=args.normal, partition='train', num_points=args.num_points, transform=train_transforms),
                                num_workers=args.num_workers,   batch_size=args.batch_size,
                                shuffle=True,   drop_last=True)
    test_loader = DataLoader(   ModelNet40(normal_channel=args.normal, partition='test', num_points=args.num_points), # transform=test_transforms),
                                num_workers=args.num_workers,   batch_size=args.batch_size,
                                shuffle=False,   drop_last=False)

    # --- Create Model
    selector = RandomNegativeTripletSelector(0.3, cpu=False, dist_func=pdist_pc)
    # selector = HardestNegativeTripletSelector(0.3, cpu=False, dist_func=pdist_pc)
    criterion1 = OnlineTripletLoss(   margin = 0.3,   # triplet loss
                                    triplet_selector = selector,
                                    dist_func = chamfer_pc)
    pointfield = PF_Tnet(dim=64, tri_criterion=criterion1)
    # pointfield = PF_Tnet(dim=64)  # pointfield
    classifier = Pointnet2(num_class=40, normal_channel=args.normal).to(device)
    criterion = get_loss  # 分类器 loss
    pf_path = BASE_DIR + '\\logs\\pointfield_margin03_pointnet\\checkpoints\\recent_pointfield.t7'
    comb_model = PF_Model(classifier, pointfield, criterion,
                    use_pointfield=args.use_pointfield, train_pointfield=True,
                ).to(device)
    # --- Optimizer
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            comb_model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(comb_model.parameters(), lr=args.lr*100,
            momentum=0.9, weight_decay=args.decay_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.6)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=200, eta_min=0.0005)  # SGDR

    trainer = Trainer(train_loader, test_loader, comb_model, optimizer, scheduler,
                        num_epochs=args.num_epochs, exp_name=args.exp_name,
                        log_dir=args.log_dir, train_file=__file__, track_grid=True)

    trainer.logger.cprint('PARAMETERS...')
    trainer.logger.cprint(args)

    trainer.train()

if __name__ == "__main__":
    main()