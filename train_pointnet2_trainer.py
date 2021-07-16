#!/usr/bin/env python
# -*- coding: utf-8 -*-
import shutil
import argparse
import os.path as osp
import numpy as np
from tqdm import tqdm
import sklearn.metrics as metrics
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.data_utils import ModelNet40, pointnet_train_transforms, pointnet_test_transforms

from models.pointnet2_cls_ssg import get_model, get_loss

from pointfield.model import CombinedModel
from pointfield.train_utils import Trainer

BASE_DIR = '/'.join(osp.abspath(__file__).split('\\')[0:-1])

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--model',          type=str,   default='pointnet2',    help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--exp_name',       type=str,   default='pointnet2_nopf',   help='expriment name')
    parser.add_argument('--log_dir',        type=str,   default='logs',     help='log directory')
    parser.add_argument('--batch_size',     type=int,   default=72,     help='batch size in training [default: 24]')
    parser.add_argument('--num_points',     type=int,   default=1024,   help='Point Number [default: 1024]')
    parser.add_argument('--num_epochs',     type=int,   default=200,    help='number of epoch in training [default: 200]')
    parser.add_argument('--num_workers',    type=int,   default=4,      help='Worker Number [default: 8]')
    parser.add_argument('--optimizer',      type=str,   default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--lr',             type=float, default=0.001,  help='learning rate in training [default: 0.001, 0.1 if using sgd]')
    parser.add_argument('--normal',         type=bool,  default=True,   help='Whether to use normal information [default: True]')
    parser.add_argument('--seed',           type=int,   default=1,      help='random seed [efault: 1]')
    parser.add_argument('--decay_rate',     type=float, default=1e-4,   help='decay rate [default: 1e-4]')
    # pointfield
    parser.add_argument('--use_pointfield', type=bool,   default=False,         metavar='N', help='Num of nearest neighbors to use')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    return args



def main():
    #--- Training settings
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # --- DataLoader
    print('Load dataset...')
    train_transforms = pointnet_train_transforms
    test_transforms = pointnet_test_transforms
    train_loader = DataLoader(  ModelNet40(normal_channel=args.normal, partition='train', num_points=args.num_points, transform=train_transforms),
                                num_workers=args.num_workers,   batch_size=args.batch_size,
                                shuffle=True,   drop_last=True)
    test_loader = DataLoader(   ModelNet40(normal_channel=args.normal, partition='test', num_points=args.num_points, transform=test_transforms),
                                num_workers=args.num_workers,   batch_size=args.batch_size,
                                shuffle=False,   drop_last=False)
    device = torch.device("cuda" if args.cuda else "cpu")

    # --- Create Model
    classifier = get_model(num_class=40, normal_channel=args.normal).to(device)
    criterion = get_loss
    comb_model = CombinedModel(classifier, use_pointfield=args.use_pointfield).to(device)
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
        optimizer = torch.optim.SGD(comb_model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.6)

    trainer = Trainer(train_loader, test_loader, comb_model, criterion, optimizer, scheduler,
                        num_epochs=args.num_epochs, exp_name=args.exp_name, log_dir=args.log_dir, train_file=__file__)

    trainer.logger.cprint('PARAMETERS...')
    trainer.logger.cprint(args)

    trainer.train()

if __name__ == "__main__":
    main()