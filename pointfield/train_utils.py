import logging
import numpy as np
import shutil
import os
import os.path as osp
from tqdm import tqdm
import torch
import sklearn.metrics as metrics

class LogTool():
    def __init__(self, file_path, format=None):
        self.logger = logging.getLogger("Default")
        self.logger.setLevel(logging.INFO)

        if format:
            formatter = logging.Formatter(format)
        else:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def cprint(self, text):
        print(text)
        self.logger.info(text)


def log_init( exp_name, trainfile):
    filepath = os.path.abspath(trainfile)
    filename = filepath.split('\\')[-1]
    root_dir = '/'.join(filepath.split('\\')[0:-1])
    log_dir = root_dir + '/logs/' + exp_name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(log_dir+'/checkpoints'):
        os.makedirs(log_dir+'/checkpoints')
    shutil.copy(trainfile, log_dir+'/'+filename+'.backup')
    print('Log file: %s' % log_dir + '/run.log')
    return LogTool(log_dir + '/run.log')


class Trainer():
    def __init__(   self, train_loader, test_loader, model, criterion,
                    optimizer, scheduler, num_epochs, exp_name, train_file, log_dir='logs'):
        self.train_file  = osp.abspath(train_file)
        self.exp_dir = osp.join(osp.dirname(self.train_file), log_dir, exp_name)
        self.logger = self.__log_init()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.test_loader  = test_loader
        self.model      = model.to(self.device)
        self.criterion  = criterion
        self.optimizer  = optimizer
        self.scheduler  = scheduler

        self.num_epochs = num_epochs
        self.best_inst_acc  = 0
        self.best_class_acc = 0
        self.epoch = 1

    def __log_init(self):
        flag = False
        if  osp.exists(self.exp_dir):
            print('The specified exp_dir already exists.')
            c = input('Continue? (y/[n])? ')
            if c in ['y', 'Y']:
                flag = True
        else:
            os.makedirs(self.exp_dir)
            flag = True
        if not flag:
            exit()

        if not osp.exists(self.exp_dir+'\\checkpoints'):
            os.makedirs(self.exp_dir+'\\checkpoints')
        shutil.copy(    self.train_file,
                        osp.join(self.exp_dir, osp.basename(self.train_file)+'.backup'))
        print('Log file: %s' % self.exp_dir + '\\run.log')
        return LogTool(self.exp_dir + '\\run.log')


    def train_epoch(self):
        self.model.train()
        train_pred = []
        train_true = []
        with tqdm(self.train_loader, total=len(self.train_loader), smoothing=0.9) as t:
            for data, label in t:
                data, label = data.to(self.device), label.to(self.device).squeeze()

                self.optimizer.zero_grad()
                logits = self.model(data, label)  # logits - B x classes
                cls_loss = self.criterion(logits, label)
                loss = cls_loss + self.model.tri_loss + self.model.reg_loss
                loss.backward()
                self.optimizer.step()

                preds = logits.max(dim=1)[1]  # preds - (B,)
                train_true.append(label)
                train_pred.append(preds)
                t.set_postfix(  cls_loss = cls_loss.item(),
                                tri_loss = self.model.tri_loss.item(),
                                reg_loss = self.model.reg_loss.item())

        self.scheduler.step()
        train_true = torch.cat(train_true)
        train_pred = torch.cat(train_pred)
        train_inst_acc = self.__accuracy_score( train_true, train_pred)
        self.logger.cprint('Train Instance Accuracy: %f' % train_inst_acc)


    def eval_epoch(self):
        self.model.eval()
        test_pred = []
        test_true = []
        with torch.no_grad():
            for data, label in tqdm(self.test_loader, total=len(self.test_loader)):
                data, label = data.to(self.device), label.to(self.device).squeeze()
                logits = self.model(data)
                preds = logits.max(dim=1)[1]
                test_true.append(label)
                test_pred.append(preds)

        test_true = torch.cat(test_true)
        test_pred = torch.cat(test_pred)
        test_inst_acc = self.__accuracy_score(test_true, test_pred)
        test_class_acc = self.__balanced_accuracy_score(test_true, test_pred)
        self.logger.cprint('Test Instance Accuracy: %f, Class Accuracy: %f'% (test_inst_acc, test_class_acc))
        if test_class_acc > self.best_class_acc:
            self.best_class_acc = test_class_acc
        if test_inst_acc >= self.best_inst_acc:
            self.best_inst_acc = test_inst_acc
            self.save_checkpoint('best_model.t7')
        self.logger.cprint('Best Instance Accuracy: %f, Class Accuracy: %f'% (self.best_inst_acc, self.best_class_acc))


    def train(self):
        self.load_checkpoint('recent_model.t7')
        self.logger.cprint('Start training...')
        while self.epoch <= self.num_epochs:
            self.logger.cprint('Epoch %d/%s:' % (self.epoch, self.num_epochs))
            self.train_epoch()
            self.eval_epoch()
            if self.epoch%10 == 0:
                self.save_checkpoint("recent_model.t7")
            self.epoch += 1
        self.logger.cprint('End of training...')


    def eval(self, model_file):
        self.logger.cprint('Start evaluate %s...' % model_file)
        if self.load_checkpoint(model_file):
            self.eval_epoch()


    def load_checkpoint(self, filename):
        try:
            checkpoint = torch.load(osp.join(self.exp_dir, "checkpoints", filename))
            self.epoch = checkpoint['epoch']
            self.best_inst_acc = checkpoint['best_inst_acc']
            self.best_class_acc = checkpoint['best_class_acc']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.logger.cprint('Load pretrain model')
            return True
        except:
            self.logger.cprint('No existing model')
            return False


    def save_checkpoint(self, filename):
        save_file = osp.join(self.exp_dir, "checkpoints", filename)
        state = {
            'epoch': self.epoch,
            'best_inst_acc': self.best_inst_acc,
            'best_class_acc': self.best_class_acc,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        torch.save(state, save_file)
        self.logger.cprint('Saving model at %s' % save_file)


    def __accuracy_score(self, y_true, y_pred):
        if len(y_pred) != len(y_true):
            raise ValueError
        return float((y_pred==y_true).sum()) / len(y_pred)


    def __balanced_accuracy_score(self, y_true, y_pred):
        if len(y_pred) != len(y_true):
            raise ValueError
        class_acc = []
        for cat in y_true.unique():
            index = (y_true==cat)
            if index.sum():
                acc = (y_pred[index]==y_true[index]).sum() / index.sum()
                class_acc.append(acc.item())
            else:
                class_acc.append(0)
        return np.mean(class_acc)



