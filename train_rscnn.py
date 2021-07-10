#!/usr/bin/env python
# -*- coding: utf-8 -*-
import shutil
import argparse
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
from tqdm import tqdm

from data.data_utils import ModelNet40, rscnn_transforms
from model import PointNet, DGCNN
from util import cal_loss

from torch.optim.lr_scheduler import CosineAnnealingLR
from pointfield.model import CombinedModel
from pointfield.train_utils import log_init

BASE_DIR = '/'.join(osp.abspath(__file__).split('\\')[0:-1])

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser(description='Relation-Shape CNN Pointcloud Classification Training')
    parser.add_argument('--exp_name',   type=str,   default='exp_rscnn',    help='Name of the experiment')
    parser.add_argument('--num_workers',type=int,   default=8,              help='num of workers')
    parser.add_argument('--batch_size', type=int,   default=32,             help='Size of batch')
    parser.add_argument('--epochs',     type=int,   default=200,            help='number of episode to train ')
    parser.add_argument('--normal',     type=bool,  default=False,          help='Whether to use normal information [default: False]')
    parser.add_argument('--seed',       type=int,   default=1,              help='random seed (default: 1)')
    parser.add_argument('--cuda',       type=bool,  default=True,           help='enables CUDA training')

    parser.add_argument('--use_sgd',    type=bool,  default=True,       help='Use SGD')
    parser.add_argument('--lr',         type=float, default=0.001,      metavar='LR',
                                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum',   type=float, default=0.9,        metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda',    type=bool,  default=False,      help='enables CUDA training')
    parser.add_argument('--seed',       type=int,   default=1,          metavar='S', help='random seed (default: 1)')
    parser.add_argument('--eval',       type=bool,  default=False,      help='evaluate the model')
    parser.add_argument('--num_points', type=int,   default=1024,       help='num of points to use')
    parser.add_argument('--dropout',    type=float, default=0.5,        help='dropout rate')
    parser.add_argument('--emb_dims',   type=int,   default=1024,       metavar='N', help='Dimension of embeddings')
    parser.add_argument('--k',          type=int,   default=20,         metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str,   default='',         metavar='N', help='Pretrained model path')

    return parser.parse_args()


def main():
    #--- Training settings
    args = parse_args()
    logger = log_init(exp_name=args.exp_name, trainfile=__file__)
    logger.cprint('PARAMETERS...')
    logger.cprint(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")

    # --- DataLoader
    print('Load dataset...')

    train_loader = DataLoader(  ModelNet40(partition='train', num_points=args.num_points, transform=rscnn_transforms),
                                num_workers=args.num_workers,   batch_size=args.batch_size,
                                shuffle=True,   drop_last=True)
    test_loader = DataLoader(   ModelNet40(partition='test', num_points=args.num_points),
                                num_workers=args.num_workers,   batch_size=args.batch_size,
                                shuffle=False,   drop_last=False)

    # --- Create Model
    classifier = RSCNN_SSN(num_classes = args.num_classes, input_channels = args.input_channels, relation_prior = args.relation_prior, use_xyz = True)

    criterion = $xxx$
    comb_model = CombinedModel(classifier)

    # --- Optimizer
    optimizer = $xxx$
    scheduler = $xxx$

    # --- Load Checkpoint
    try:
        checkpoint = torch.load(BASE_DIR+'logs/%s/checkpoints/model.t7' % args.exp_name)
        start_epoch = checkpoint['epoch']
        best_inst_acc = checkpoint['inst_acc']
        best_class_acc = checkpoint['class_acc']
        comb_model.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        comb_model.pointfield.load_state_dict(checkpoint['pointfield_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.cprint('Use pretrain model')
    except:
        logger.cprint('No existing model, starting training from scratch...')
        start_epoch = 0
        best_inst_acc = 0
        best_class_acc = 0

    # --- train
    logger.cprint('Start training...')
    for epoch in range(start_epoch, args.epochs):
        logger.cprint('Epoch %d/%s:' % (epoch + 1, args.epochs))

        # train epoch
        train_pred = []
        train_true = []
        with tqdm(train_loader, total=len(train_loader), smoothing=0.9) as t:
            comb_model.train()

            for data, label in t:
                data, label = data.to(device), label.to(device).squeeze()

                # if torch.unique(label).nelement() < 3 or torch.unique(label).nelement() == args.batch_size:
                #     continue

                optimizer.zero_grad()
                preds = comb_model(data, label)  ## preds - B x 1
                cls_loss = criterion(preds, label)
                loss = cls_loss + comb_model.tri_loss + comb_model.reg_loss
                loss.backward()
                optimizer.step()

                preds = preds.max(dim=1)[1]
                train_true.append(label.cpu().numpy())
                train_pred.append(preds.detach().cpu().numpy())
                t.set_postfix(  cls_loss = cls_loss.item(),
                                tri_loss = comb_model.tri_loss.item(),
                                reg_loss = comb_model.reg_loss.item())
        scheduler.step()
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_inst_acc = metrics.accuracy_score( train_true, train_pred)
        logger.cprint('Train Instance Accuracy: %f' % train_inst_acc)

        # test epoch
        test_pred = []
        test_true = []
        with torch.no_grad():
            comb_model.eval()
            for data, label in tqdm(test_loader, total=len(test_loader)):
                data, label = data.to(device), label.to(device).squeeze()
                preds = comb_model(data)
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_inst_acc = metrics.accuracy_score(test_true, test_pred)
        test_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)

        # Save Checkpoint
        save_dir = BASE_DIR+'logs/%s/checkpoints/' % args.exp_name
        if (test_class_acc > best_class_acc):
            best_class_acc = test_class_acc
        if test_inst_acc>=best_inst_acc or (epoch+1)%10==0:
            logger.cprint('Saving model at %smodel.t7' % save_dir)
            if test_inst_acc >= best_inst_acc:
                best_inst_acc = test_inst_acc
            state = {
                'epoch': epoch + 1,
                'inst_acc': test_inst_acc,
                'class_acc': test_class_acc,
                'classifier_state_dict': comb_model.classifier.state_dict(),
                'pointfield_state_dict': comb_model.pointfield.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }
            torch.save(state, save_dir+'model.t7')
            if test_inst_acc >= best_inst_acc:
                logger.cprint('Saving best model at %sbest_model.t7' % save_dir)
                shutil.copy(save_dir+'model.t7', save_dir+'best_model.t7')

        logger.cprint('Test Instance Accuracy: %f, Class Accuracy: %f'% (test_inst_acc, test_class_acc))
        logger.cprint('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_inst_acc, best_class_acc))
    logger.cprint('End of training...')


if __name__ == "__main__":
    main()