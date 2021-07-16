#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os.path as osp
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

import data.data_utils as dutil
from data.data_utils import ModelNet40

from pointfield.model import PointField, get_loss
from pointfield.train_utils import Trainer

BASE_DIR = '/'.join(osp.abspath(__file__).split('\\')[0:-1])

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser(description='DGCNN')
    parser.add_argument('--model',          type=str,   default='pointfield',    help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--exp_name',       type=str,   default='pointfield',   help='expriment name')
    parser.add_argument('--log_dir',        type=str,   default='logs',     help='log directory')
    parser.add_argument('--batch_size',     type=int,   default=72,     help='batch size in training [default: 24]')
    parser.add_argument('--num_points',     type=int,   default=1024,   help='Point Number [default: 1024]')
    parser.add_argument('--num_epochs',     type=int,   default=250,    help='number of epoch in training [default: 200]')
    parser.add_argument('--num_workers',    type=int,   default=3,      help='Worker Number [default: 8]')
    parser.add_argument('--optimizer',      type=str,   default='SGD', help='optimizer for training [default: Adam]')
    parser.add_argument('--normal',         type=bool,  default=True,   help='Whether to use normal information [default: True]')
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
    train_transforms = transforms.Compose([ dutil.translate_pointcloud,
                                            dutil.shuffle_pointcloud])

    print('Load dataset...')
    train_loader = DataLoader(  ModelNet40(partition='train', num_points=args.num_points, transform=train_transforms),
                                num_workers=args.num_workers,   batch_size=args.batch_size,
                                shuffle=True,   drop_last=True)
    test_loader = DataLoader(   ModelNet40(partition='test', num_points=args.num_points),
                                num_workers=args.num_workers,   batch_size=args.batch_size,
                                shuffle=False,   drop_last=False)

    # --- Create Model
    pointfield = PointField(64).to(device)
    criterion = get_loss

    # --- Optimizer
    if args.optimizer == 'SGD':
        print("Use SGD")
        optimizer = torch.optim.SGD(pointfield.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        optimizer = torch.optim.Adam(pointfield.parameters(), lr=args.lr, weight_decay=1e-4 )

    scheduler = CosineAnnealingLR(optimizer, args.num_epochs, eta_min=args.lr)

    trainer = Trainer(train_loader, test_loader, pointfield, criterion, optimizer, scheduler,
                        num_epochs=args.num_epochs, exp_name=args.exp_name, log_dir=args.log_dir, train_file=__file__)

    trainer.logger.cprint('PARAMETERS...')
    trainer.logger.cprint(args)

    trainer.train_pointfield()

if __name__ == "__main__":
    main()