import numpy as np
import shutil
import os
import os.path as osp
from tqdm import tqdm
import torch
from time import time
from .visualize_utils import draw_grid
from .log_tool import LogTool
import cv2

class Trainer():

    def __init__(   self, train_loader, test_loader, model, optimizer, scheduler,
            num_epochs, train_file, exp_name, log_dir='logs', track_grid=False):
        """训练类

        Args:
            train_loader : 训练集
            test_loader : 测试集
            model (PF_Model): 模型
            optimizer : 优化器
            scheduler : 规划器
            num_epochs (int): 迭代次数
            train_file (str): 训练文件, 将被保存在实验目录中
            exp_name (str): 实验名称, 实验目录为 'log_dir/exp_name/'
            log_dir (str, optional): log文件目录. Defaults to 'logs'.
            track_grid (bool, optional): 是否保存pointfield的grid可视化结果. Defaults to False.
        """
        #　日志设置
        self.train_file  = osp.abspath(train_file)
        self.exp_dir = osp.join(osp.dirname(self.train_file), log_dir, exp_name)  # 实验目录
        self.logger = self.__log_init()
        self.track_grid = track_grid  # 是否保存 pointfield grid 图像 TODO: 目前只能用于单层pointfield

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.test_loader  = test_loader
        self.model      = model.to(self.device)
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        # 训练过程
        self.num_epochs = num_epochs
        self.best_inst_acc  = 0
        self.best_class_acc = 0
        self.curr_epoch = 1  # 当前epoch


    def __log_init(self):
        """初始化实验日志

        Returns:
            LogTool: 实验日志记录工具
        """
        flag = False
        if  osp.exists(self.exp_dir):
            print('The specified exp_dir already exists.')
            c = input('Continue?\n[Y] Yes  [N] No  [R] Remove exp_dir and Restart (default is "N"):')
            if c in ['N', 'n']:
                exit()
            elif c in ['R', 'r']:
                shutil.rmtree(self.exp_dir)
                os.makedirs(self.exp_dir)
        else:
            os.makedirs(self.exp_dir)

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
        with tqdm(self.train_loader, total=len(self.train_loader), ncols=100, smoothing=0.9) as t:
            for data, label in t:
                data, label = data.to(self.device), label.to(self.device).squeeze()

                self.optimizer.zero_grad()
                logits, tri_loss, reg_loss, cls_loss = self.model(data, label)  # logits - B x classes
                loss = cls_loss + tri_loss + reg_loss
                loss.backward()
                self.optimizer.step()

                preds = logits.max(dim=1)[1]  # logits每行最大值的下标为预测类别, preds - (B,)
                train_true.append(label)
                train_pred.append(preds)
                t.set_postfix(  cls_loss = cls_loss.item(),  # 进度条后缀
                                tri_loss = tri_loss.item(),
                                reg_loss = reg_loss.item())

        self.scheduler.step()
        train_true = torch.cat(train_true)
        train_pred = torch.cat(train_pred)
        train_inst_acc = self.__accuracy_score( train_true, train_pred)  #　计算本 epoch 分类准确率
        self.logger.cprint('Train Instance Accuracy: {:f}, loss: cls={:f}, tri={:f}, reg={:f}'.format(
                            train_inst_acc, cls_loss, tri_loss, reg_loss))


    def eval_epoch(self):
        self.model.eval()
        test_pred = []
        test_true = []
        with torch.no_grad():
            for data, label in tqdm(self.test_loader, total=len(self.test_loader), ncols=100):
                data, label = data.to(self.device), label.to(self.device).squeeze()
                logits = self.model(data, label)
                preds = logits.max(dim=1)[1]
                test_true.append(label)
                test_pred.append(preds)

        test_true = torch.cat(test_true)
        test_pred = torch.cat(test_pred)
        test_inst_acc = self.__accuracy_score(test_true, test_pred)  # 计算分类准确率
        test_class_acc = self.__balanced_accuracy_score(test_true, test_pred)  # 计算均衡分类准确率
        self.logger.cprint('Test Instance Accuracy: %f, Class Accuracy: %f'% (test_inst_acc, test_class_acc))
        if test_class_acc > self.best_class_acc:  # 更新最高均衡类别准确率
            self.best_class_acc = test_class_acc
        if test_inst_acc >= self.best_inst_acc:  # 如果测试分类准确率大于最高值, 更新最高准确率, 并保存模型检查点
            self.best_inst_acc = test_inst_acc
            self.save_checkpoint('best_model.t7')
            self.save_checkpoint('recent_model.t7')
        self.logger.cprint('Best Instance Accuracy: %f, Class Accuracy: %f'% (self.best_inst_acc, self.best_class_acc))


    def train(self):
        t0 = time()
        self.load_checkpoint('recent_model.t7')  # 尝试读取预训练模型
        self.logger.cprint('Start training...')
        while self.curr_epoch <= self.num_epochs:
            t1 = time()
            self.logger.cprint('Epoch %d/%s:' % (self.curr_epoch, self.num_epochs))
            self.train_epoch()
            self.eval_epoch()
            if self.curr_epoch%10==0:  # 每10个epoch保存一次模型
                self.save_checkpoint("recent_model.t7")
            # --- track grid TODO
            if self.track_grid and (self.curr_epoch%20==0 or self.curr_epoch==1 or self.curr_epoch==self.num_epochs):
                grid_img = self.model.pointfield.draw_grid()
                cv2.imwrite(filename=self.exp_dir+'\\checkpoints\\epoch_{:0>3d}.jpg'.format(self.curr_epoch),
                            img=grid_img)

            self.curr_epoch += 1
            self.logger.info('Epoch elapsed time: ' + self.__format_second(time()-t1))
        self.logger.cprint('Total time: ' + self.__format_second(time()-t0))
        self.logger.cprint('End of training...')


    def eval(self, model_file):
        """ 测试预训练模型 """
        self.logger.cprint('Start evaluate %s...' % model_file)
        if self.load_checkpoint(model_file):  #　当成功加载到预训练模型时
            self.eval_epoch()


    def load_checkpoint(self, filename):
        try:  # 从'实验目录/checkpoints' 中读取预训练模型
            checkpoint = torch.load(osp.join(self.exp_dir, "checkpoints", filename))
            self.curr_epoch = checkpoint['curr_epoch'] + 1
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
            'curr_epoch': self.curr_epoch,
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


    def __format_second(self, time):
        mins, s = divmod(int(time), 60)
        h, m = divmod(mins, 60)
        if h:
            return '{0:d}:{1:02d}:{2:02d}'.format(h, m, s)
        else:
            return '{0:02d}:{1:02d}'.format(m, s)


    def train_pointfield(self):
        """ 单独训练 pointfield """
        t0 = time()
        self.best_class_acc = torch.tensor(10000)
        self.load_checkpoint('recent_pointfield.t7')
        self.logger.cprint('Start training pointfield...')
        while self.curr_epoch <= self.num_epochs:
            t1 = time()
            self.logger.cprint('Epoch %d/%s:' % (self.curr_epoch, self.num_epochs))
            self.train_epoch_pointfield()
            self.eval_epoch_pointfield()
            if self.curr_epoch%10==0:
                self.save_checkpoint("recent_pointfield.t7")

            if self.track_grid and (self.curr_epoch%20==0 or self.curr_epoch==1 or self.curr_epoch==self.num_epochs):
                grid_img = self.model.draw_grid()
                cv2.imwrite(filename=self.exp_dir+'\\checkpoints\\epoch_{:0>3d}.jpg'.format(self.curr_epoch),
                            img=grid_img)

            self.curr_epoch += 1
            self.logger.info('Epoch elapsed time: ' + self.__format_second(time()-t1))
        self.logger.cprint('Total time: ' + self.__format_second(time()-t0))
        self.logger.cprint('End of training...')


    def train_epoch_pointfield(self):
        self.model.train()

        with tqdm(self.train_loader, total=len(self.train_loader), ncols=100, smoothing=0.9) as t:
            track_tri_loss = torch.zeros(len(self.train_loader))
            i = 0
            for data, label in t:
                data, label = data.to(self.device), label.to(self.device).squeeze()
                self.optimizer.zero_grad()
                data, tri_loss, reg_loss = self.model(data, label, require_loss=True)
                loss = tri_loss + reg_loss
                loss.backward()
                self.optimizer.step()

                track_tri_loss[i] = tri_loss
                i += 1
                t.set_postfix(tri_loss=tri_loss.item(), reg_loss=reg_loss.item())

        self.scheduler.step()
        mean_tri_loss = track_tri_loss.mean()
        self.logger.cprint('mean triplet loss: {:f}'.format(mean_tri_loss.item()))


    def eval_epoch_pointfield(self):
        self.model.eval()
        i = 0
        track_tri_loss = torch.zeros(len(self.test_loader))
        with torch.no_grad():
            with tqdm(self.test_loader, total=len(self.test_loader), ncols=100) as t:
                for data, label in t:
                    data, label = data.to(self.device), label.to(self.device).squeeze()
                    data, tri_loss, reg_loss = self.model(data, label, require_loss=True)
                    track_tri_loss[i] = tri_loss
                    i += 1
                    t.set_postfix(tri_loss=tri_loss.item(), reg_loss=reg_loss.item())

        mean_tri_loss = track_tri_loss.mean()
        self.logger.cprint('test mean triplet loss: {:f}'.format(mean_tri_loss))
        if mean_tri_loss < self.best_class_acc:
            self.best_class_acc = mean_tri_loss
            self.save_checkpoint('best_pointfield.t7')
            self.save_checkpoint('recent_pointfield.t7')
        self.logger.cprint('Best mean triplet loss: %f'% (self.best_class_acc))

