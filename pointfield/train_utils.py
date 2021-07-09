import logging
import numpy as np
import shutil
import os

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
    filename = trainfile.split('/')[-1]
    root_dir = '/'.join(trainfile.split('/')[0:-1])
    log_dir = root_dir + '/logs/' + exp_name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(log_dir+'/checkpoints'):
        os.makedirs(log_dir+'/checkpoints')
    shutil.copy(trainfile, log_dir+'/'+filename+'.backup')
    print('Log file: %s' % log_dir + '/run.log')
    return LogTool(log_dir + '/run.log')

def accuracy_score(y_true, y_pred):
    if len(y_pred) != len(y_true):
        raise ValueError
    return float((y_pred==y_true).sum()) / len(y_pred)

def class_accuracy_score(y_true, y_pred):
    if len(y_pred) != len(y_true):
        raise ValueError
    class_acc = []
    for cat in set(y_pred):
        index = (y_true==cat)
        acc = (y_pred[index]==y_true[index]).sum() / float(index.sum())
        class_acc.append(acc)
    return np.mean(class_acc)