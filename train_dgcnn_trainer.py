#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os.path as osp
import shutil
import argparse
import numpy as np
from torch._C import CudaIntStorageBase
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import sklearn.metrics as metrics

from data.data_utils import ModelNet40, dgcnn_transforms
from models.dgcnn import PointNet, DGCNN, get_loss

from pointfield.model import CombinedModel
from pointfield.train_utils import Trainer

BASE_DIR = '/'.join(osp.abspath(__file__).split('\\')[0:-1])

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--model',      type=str,   default='dgcnn',    help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--exp_name',   type=str,   default='dgcnn_exp1',      metavar='N', help='Name of the experiment')
    parser.add_argument('--num_points', type=int,   default=1024,       help='num of points to use')
    parser.add_argument('--log_dir',    type=str,   default='logs',     help='log directory')
    parser.add_argument('--batch_size', type=int,   default=32,         metavar='batch_size', help='Size of batch')
    parser.add_argument('--num_epochs', type=int,   default=250,        metavar='N', help='number of episode to train ')
    parser.add_argument('--use_sgd',    type=bool,  default=False,      help='Use SGD')
    parser.add_argument('--lr',         type=float, default=0.001,      help='min learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum',   type=float, default=0.9,        metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed',       type=int,   default=1,          metavar='S', help='random seed (default: 1)')
    parser.add_argument('--num_workers',type=int,   default=8,          metavar='N', help='num of workers')

    parser.add_argument('--dropout',    type=float, default=0.5,        help='dropout rate')
    parser.add_argument('--emb_dims',   type=int,   default=1024,       metavar='N', help='Dimension of embeddings')
    parser.add_argument('--k',          type=int,   default=20,         metavar='N', help='Num of nearest neighbors to use')
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
    train_loader = DataLoader(  ModelNet40(partition='train', num_points=args.num_points, transform=dgcnn_transforms),
                                num_workers=args.num_workers,   batch_size=args.batch_size,
                                shuffle=True,   drop_last=True)
    test_loader = DataLoader(   ModelNet40(partition='test', num_points=args.num_points),
                                num_workers=args.num_workers,   batch_size=args.batch_size,
                                shuffle=False,   drop_last=False)

    # --- Create Model
    device = torch.device("cuda" if args.cuda else "cpu")
    classifier = DGCNN(args).to(device)
    criterion = get_loss
    comb_model = CombinedModel(classifier).to(device)

    # --- Optimizer
    if args.use_sgd:
        print("Use SGD")
        optimizer = optim.SGD(
            comb_model.parameters(),
            lr=args.lr*100,
            momentum=args.momentum,
            weight_decay=1e-4
        )
    else:
        print("Use Adam")
        optimizer = optim.Adam(
            comb_model.parameters(),
            lr=args.lr,
            weight_decay=1e-4
        )
    scheduler = CosineAnnealingLR(optimizer, args.num_epochs, eta_min=args.lr)

    trainer = Trainer(train_loader, test_loader, comb_model, criterion, optimizer, scheduler,
                        num_epochs=args.num_epochs, exp_name=args.exp_name, log_dir=args.log_dir, train_file=__file__)

    trainer.logger.cprint('PARAMETERS...')
    trainer.logger.cprint(args)

    trainer.train()

    # --- Evaluate
    # trainer.eval('best_model.t7')

if __name__ == "__main__":
    main()