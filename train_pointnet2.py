#!/usr/bin/env python
# -*- coding: utf-8 -*-
import shutil
import argparse
import numpy as np
from tqdm import tqdm
import sklearn.metrics as metrics
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.data_utils import ModelNet40, pointnet_train_transforms, pointnet_test_transforms
import data.data_utils as dutil

from models.pointnet2_cls_ssg import get_model, get_loss

from pointfield.model import CombinedModel
from pointfield.train_utils import log_init

BASE_DIR = '/'.join(__file__.split('/')[0:-1])

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size',     default=24,     type=int,   help='batch size in training [default: 24]')
    parser.add_argument('--exp_name',       default='pointnet2_exp',    help='expriment name [default: pointnet2_exp]')
    parser.add_argument('--epochs',         default=200,    type=int,   help='number of epoch in training [default: 200]')
    parser.add_argument('--learning_rate',  default=0.001,  type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--cuda',           default=True,   type=bool,  help='use cuda [default: cuda]')
    parser.add_argument('--num_points',     default=1024,   type=int,   help='Point Number [default: 1024]')
    parser.add_argument('--num_workers',    default=8,      type=int,   help='Worker Number [default: 8]')
    parser.add_argument('--optimizer',      default='Adam', type=str,   help='optimizer for training [default: Adam]')
    parser.add_argument('--decay_rate',     default=1e-4,   type=float, help='decay rate [default: 1e-4]')
    parser.add_argument('--normal',         default=True,   action='store_true', help='Whether to use normal information [default: True]')
    parser.add_argument('--seed',           default=1,      type=int,   help='random seed (default: 1)')
    return parser.parse_args()

def main():
    #--- Training settings
    args = parse_args()
    logger = log_init(exp_name=args.exp_name, trainfile=__file__)
    logger.cprint('PARAMETERS...')
    logger.cprint(args)
    #
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
    criterion = get_loss()
    comb_model = CombinedModel(classifier).to(device)

    # --- Optimizer
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            comb_model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(comb_model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.6)

    # --- Load Checkpoint
    try:
        checkpoint = torch.load(BASE_DIR+'/logs/%s/checkpoints/model.t7' % args.exp_name)
        start_epoch = checkpoint['epoch']
        best_inst_acc = checkpoint['inst_acc']
        best_class_acc = checkpoint['class_acc']
        comb_model.models['cls'].load_state_dict(checkpoint['classifier_state_dict'])
        comb_model.models['pf'].load_state_dict(checkpoint['pointfield_state_dict'])
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
                preds, trans_feat = comb_model(data, label)  ##
                cls_loss = criterion(preds, label)
                loss = cls_loss + comb_model.tri_loss + comb_model.reg_loss
                loss.backward()
                optimizer.step()

                preds = preds.max(dim=1)[1]  ## preds - B x 1
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
                preds, _ = comb_model(data)
                preds = preds.max(dim=1)[1]
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_inst_acc = metrics.accuracy_score(test_true, test_pred)
        test_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)

        # Save Checkpoint
        save_dir = BASE_DIR+'/logs/%s/checkpoints/' % args.exp_name
        if (test_class_acc > best_class_acc):
            best_class_acc = test_class_acc
        if test_inst_acc>=best_inst_acc or (epoch+1)%10==0:
            logger.cprint('Saving model at %smodel.t7' % save_dir)
            if test_inst_acc >= best_inst_acc:
                best_inst_acc = test_inst_acc
            state = {
                'epoch': epoch + 1,
                'inst_acc': best_inst_acc,
                'class_acc': best_class_acc,
                'classifier_state_dict': comb_model.models['cls'].state_dict(),
                'pointfield_state_dict': comb_model.models['pf'].state_dict(),
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