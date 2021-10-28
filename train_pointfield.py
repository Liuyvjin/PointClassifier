#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

import os.path as osp
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

import data.data_utils as dutil
from data.data_utils import ModelNet40, pointnet_train_transforms, pointnet_test_transforms

from pointfield import PF_1layer, PF_Nlayer, PF_Model, PF_Tnet
from pointfield import Trainer
from pointfield.losses import OnlineTripletLoss
from pointfield.loss_utils import HardestNegativeTripletSelector, pdist_pc, chamfer_pc, RandomNegativeTripletSelector

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser(description='pointfield')
    parser.add_argument('--model',          type=str,   default='pointfield',    help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--exp_name',       type=str,   default='pf_chamfer/pf_tnet_random',   help='expriment name')
    parser.add_argument('--log_dir',        type=str,   default='new_logs',     help='log directory')
    parser.add_argument('--batch_size',     type=int,   default=32,     help='batch size in training [default: 24]')
    parser.add_argument('--num_points',     type=int,   default=1024,   help='Point Number [default: 1024]')
    parser.add_argument('--num_epochs',     type=int,   default=200,    help='number of epoch in training [default: 200]')
    parser.add_argument('--num_workers',    type=int,   default=3,      help='Worker Number [default: 8]')
    parser.add_argument('--optimizer',      type=str,   default='SGD', help='optimizer for training [default: Adam]')
    parser.add_argument('--normal',         type=bool,  default=False,   help='Whether to use normal information [default: True]')
    parser.add_argument('--seed',           type=int,   default=1,      help='random seed [efault: 1]')
    parser.add_argument('--lr',             type=float, default=0.001,  help='learning rate in training [default: 0.001, 0.1 if using sgd]')
    parser.add_argument('--momentum',       type=float, default=0.9,        metavar='M', help='SGD momentum (default: 0.9)')

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
    print('Loading dataset...')
    train_transforms = transforms.Compose([ dutil.translate_pointcloud,
                                            dutil.shuffle_pointcloud])
    # test_transforms = pointnet_test_transforms
    train_loader = DataLoader(  ModelNet40(normal_channel=args.normal, partition='train', num_points=args.num_points, transform=train_transforms),
                                num_workers=args.num_workers,   batch_size=args.batch_size,
                                shuffle=True,   drop_last=True)
    test_loader = DataLoader(   ModelNet40(normal_channel=args.normal, partition='test', num_points=args.num_points),
                                num_workers=args.num_workers,   batch_size=args.batch_size,
                                shuffle=False,   drop_last=False)

    # --- Create Model
    selector = RandomNegativeTripletSelector(0.3, cpu=False, dist_func=pdist_pc)
    # selector = HardestNegativeTripletSelector(0.3, cpu=False, dist_func=pdist_pc)
    criterion = OnlineTripletLoss(   margin = 0.3,
                                    triplet_selector = selector,
                                    dist_func = chamfer_pc)

    # pointfield = PF_1layer(tri_criterion=criterion)  # 单层 pointfield
    # pointfield = PF_Nlayer(dims=[64,64,64], tri_criterion=criterion)  # 3层 pointfield
    pointfield = PF_Tnet(dim=64, tri_criterion=criterion)  # tnet pointfield

    # --- Optimizer
    if args.optimizer == 'SGD':
        print("Use SGD")
        optimizer = torch.optim.SGD(pointfield.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        optimizer = torch.optim.Adam(pointfield.parameters(), lr=args.lr, weight_decay=1e-4 )

    # scheduler = CosineAnnealingLR(optimizer, T_max=arg.num_epochs, eta_min=args.lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, eta_min=0.0001)  # SGDR

    trainer = Trainer(train_loader, test_loader, pointfield, optimizer, scheduler,
                        num_epochs=args.num_epochs, exp_name=args.exp_name,
                        log_dir=args.log_dir, train_file=__file__, track_grid=True)

    trainer.logger.cprint('PARAMETERS...')
    trainer.logger.cprint(args)

    trainer.train_pointfield()

if __name__ == "__main__":
    main()