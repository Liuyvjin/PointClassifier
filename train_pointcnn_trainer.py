#!/usr/bin/env python
# -*- coding: utf-8 -*-
import shutil
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import sklearn.metrics as metrics
from tqdm import tqdm
import os.path as osp
from data.data_utils import ModelNet40
import data.data_utils as dutil
import sched
import time
# from models.pointcnn import RandPointCNN_cls, get_loss

from pointfield.model import CombinedModel
from pointfield.train_utils import Trainer
""""""
import torch.nn.functional as F
from models.pointcnn import config
from models.pointcnn import modelnet_x3_l4


BASE_DIR = '/'.join(osp.abspath(__file__).split('\\')[0:-1])

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser(description='PointCNN')
    parser.add_argument('--model',          type=str,   default='PointCNN-fast',    help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--exp_name',       type=str,   default='PointCNN_margin03_detach',   help='expriment name')
    parser.add_argument('--log_dir',        type=str,   default='logs',     help='log directory')
    parser.add_argument('--batch_size',     type=int,   default=72,     help='batch size in training [default: 24]')
    parser.add_argument('--num_points',     type=int,   default=1024,   help='Point Number [default: 1024]')
    parser.add_argument('--num_epochs',     type=int,   default=200,    help='number of epoch in training [default: 200]')
    parser.add_argument('--num_workers',    type=int,   default=4,      help='Worker Number [default: 8]')
    parser.add_argument('--optimizer',      type=str,   default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--normal',         type=bool,  default=False,   help='Whether to use normal information [default: True]')
    parser.add_argument('--seed',           type=int,   default=1,      help='random seed [efault: 1]')
    parser.add_argument('--lr',             type=float, default=0.01,  help='learning rate in training [default: 0.001, 0.1 if using sgd]')
    parser.add_argument('--momentum',       type=float, default=0.9,    help='Initial learning rate [default: 0.9]')
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
    train_transforms = transforms.Compose([
                                                dutil.translate_pointcloud,
                                                dutil.shuffle_pointcloud])
    train_loader = DataLoader(  ModelNet40(partition='train', num_points=args.num_points, transform=train_transforms),
                                num_workers=args.num_workers,   batch_size=args.batch_size,
                                shuffle=True,   drop_last=True)
    test_loader = DataLoader(   ModelNet40(partition='test', num_points=args.num_points),
                                num_workers=args.num_workers,   batch_size=args.batch_size,
                                shuffle=False,   drop_last=False)

    # --- Create Model
    classifier = modelnet_x3_l4().to(device)
    criterion = F.cross_entropy
    pf_path = BASE_DIR + '\\logs\\pointfield_margin03_dgcnn\\checkpoints\\best_pointfield.t7'
    comb_model = CombinedModel(classifier, use_pointfield=args.use_pointfield, detach=True, pointfield_path=pf_path).to(device)

    # --- Optimizer
    if args.optimizer == 'SGD':
        print("Use SGD")
        optimizer = torch.optim.SGD(comb_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        optimizer = torch.optim.Adam(comb_model.parameters(), lr=args.lr, weight_decay=1e-4 )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    # --- Trainer
    trainer = Trainer(train_loader, test_loader, comb_model, criterion, optimizer, scheduler,
                        num_epochs=args.num_epochs, exp_name=args.exp_name, log_dir=args.log_dir, train_file=__file__)

    trainer.logger.cprint('PARAMETERS...')
    trainer.logger.cprint(args)

    trainer.train()

    # --- Evaluate
    # trainer.eval('best_model.t7')


if __name__ == "__main__":
    # main()
    scheduler = sched.scheduler(time.time, time.sleep)
    print('Waiting for start ... ')
    #分别设置在执行后n秒后执行调用函数
    scheduler.enter(500, 1, main, ())
    scheduler.run()