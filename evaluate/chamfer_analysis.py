from re import I
import sys
from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.joinpath('..').resolve()))

import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

from pointfield.loss_utils import chamfer_pc, pdist_pc
from pointfield import PF_Tnet, PF_1layer, PF_Nlayer
from data.data_utils import ModelNet40, shuffle_pointcloud


def load_sub_state_dict(sub_model, state_dict, sub_name='pointfield'):
    """ 加载模型参数到模型成员中 """
    sub_dict = {}
    pattern = '^.*' + sub_name + '\.'  # 匹配 '...pointfield.'
    pattern = re.compile(pattern, re.I)
    for key, val in state_dict.items():
        if pattern.match(key):
            new_key = pattern.sub('pf.', key)  # 去掉前缀pointfield
            sub_dict[new_key] = val
    sub_model.load_state_dict(sub_dict)

def filename(dataset, type, sub_model, savedir='exp_images'):
    suffix = '#sub_model' if sub_model else ''
    return f'{savedir}/{dataset}#{type}{suffix}.png'

def increment_path(path, exist_ok=False, sep='', mkdir=True):
    """Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    当同名path不存在时, 创建并返回
    当同名path存在时:
        若exist_ok=True: 查找相似path编号最大的一个, 创建并返回
        若exist_ok=False: 查找相似path编号最大的一个, +1 , 创建并返回

    Args:
        path (Path): 扩展前路径
        exist_ok (bool, optional): True表示不扩展, False表示要扩展. Defaults to False.
        sep (str, optional): 连接符如'_'. Defaults to ''.
        mkdir (bool, optional): 是否创建扩展目录. Defaults to True.
    Returns:
        path(Path): 扩展后的路径
    """
    path = Path(path)  # os-agnostic
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    else:  # 当前路径存在
        dirs = dir.parent.glob(f"{dir.stem}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, str(d)) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) if i else 1  # increment number, 找最大的一个
        if exist_ok:  # 使用最大的一个
            n = '' if n==1 else str(n)
            dir =  Path(f"{dir}{sep}{n}")  # update dir
        elif mkdir:  # 在 n 的基础上 +1
            dir =  Path(f"{dir}{sep}{n+1}")  # update dir
            dir.mkdir(parents=True, exist_ok=True)
    return dir

class ChamferAnalysis():
    def __init__(self, cls_nums=40) -> None:
        self.n = cls_nums
        self.idx_range = (self.n * (self.n + 1)) // 2
        self.s_cnt = np.zeros(self.idx_range, dtype=int)  # 样本数
        self.s_mean = np.zeros(self.idx_range)
        self.s_var = np.zeros(self.idx_range)

    def update(self, i, j, dist):
        """ 更新类别 i 和 j 之间的chamfer distance 均值以及方差 """
        idx = self._ij2idx(i, j)
        self.s_mean[idx], self.s_var[idx] = self.new_mean_var(dist, self.s_mean[idx],
            self.s_var[idx], self.s_cnt[idx])
        self.s_cnt[idx] += 1

    def new_mean_var(self, x, old_mean, old_var, n)->int:
        """
        均值以及方差递推公式, 已知n个样本下的均值和方差, 计算加入第 n+1 个样本后的均值和方差
        Args:
            x (float): 新的采样值
            old_mean (float): n个样本时的均值
            old_var (float): n个样本时的方差
            n (int): 已统计样本数 n >= 0
        """
        new_mean =  (n * old_mean + x) / (n + 1)  # n>=0
        dmean = old_mean - new_mean
        new_var = (n * old_var + n * dmean**2 + (x - new_mean)**2) / (n + 1)

        return new_mean, new_var

    def _ij2idx(self, i, j)->int:
        """ 将 n x n 矩阵的坐标映射到下三角, 然后拉直为一维数组的下标 """
        if i < j:
            i, j = j, i

        return (i * (i + 1)) // 2 + j

    def _idx2ij(self, idx)->int:
        i = int(np.sqrt(idx * 2)) - 2
        j = idx - (i * (i + 1)) // 2
        while j > i:
            i += 1
            j = idx - (i * (i + 1)) // 2

        return i, j

    def save(self, file=f'{BASE_DIR}/chamfer.txt'):
        """ 保存文件 """
        with open(file, 'w') as fd:
            fd.write(' i,   j,    cnt,   mean,    var\n')
            for idx in range(self.idx_range):
                i, j = self._idx2ij(idx)
                line = f'{i:2d},  {j:2d},  {self.s_cnt[idx]:5d},  {self.s_mean[idx]:5.3f},  {self.s_var[idx]:5.3f}\n'
                fd.write(line)

    def draw_heatmap(self, file=f'{BASE_DIR}/save_imgs/train#None.png', save=False, show=False):
        """ 绘制热力图 """
        heatmap = np.zeros((self.n, self.n))
        for idx in range(self.idx_range):
            i, j = self._idx2ij(idx)
            heatmap[i, j] = self.s_mean[idx]
            heatmap[j, i] = self.s_mean[idx]

        fig, ax = plt.subplots()
        im = ax.imshow(heatmap)
        ax.set_xlabel('class id')
        ax.set_ylabel('class id')
        file_strs = Path(file).stem.split('#')
        ax.set_title(f'Mean Chamfer Distance Between Classes\nDataset: {file_strs[0]},  Model: {file_strs[1]}')
        plt.colorbar(im)
        if save:
            plt.savefig(file)
        if show:
            plt.show()

    def load_model(self, type, path, device, sub_model=False):
        """ 加载模型 """
        if type == 'PF_1layer':
            self.model = PF_1layer(dim=64)
        elif type == 'PF_3layer':
            self.model = PF_Nlayer(dims=[64,64,64])
        elif type == 'PF_Tnet':
            self.model = PF_Tnet(dim=64)
        else:
            self.model = None

        if self.model is not None:
            checkpoint = torch.load(path)
            if sub_model:  # 为子模型
                checkpoint = torch.load(path)
                print(checkpoint['model_state_dict'].keys())
                load_sub_state_dict(self.model, checkpoint['model_state_dict'])
                self.model = self.model.to(device)
            else:  # 独立训练模型
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model = self.model.to(device)


# 设置
dataset = ['train', 'test']  # 分析chamfer distance的数据集
save = True  # 是否保存图片
savedir_name = 'exp_images'  # 图片保存目录
exist_ok = True  # true表示不创建新目录, false创建新目录, 默认创建新目录

show = False  # 是否显示热力图
settings = [
    {   # 固定dgcnn, 单独训练pf 1layer
        'type': 'PF_1layer',
        'sub_model': True,  # pf是否是权重文件中的子模型
        'model_path': r"D:\work\PointClassifier\logs\dgcnn_exp_sgd_nopf\checkpoints\best_model.t7"  # 权重路径
    },
    # {   # 修改 tnet 旋转矩阵计算方法后, 改用 random hardest triplet selecter
    #     'type': 'PF_Tnet',
    #     'sub_model': False,  # pf是否是权重文件中的子模型
    #     'model_path': r"D:\work\PointClassifier\new_logs\pf_chamfer\pf_tnet_random\checkpoints\best_pointfield.t7"  # 权重路径
    # },
    # {   # 修改 tnet 旋转矩阵计算方法后
    #     'type': 'PF_Tnet',
    #     'sub_model': False,  # pf是否是权重文件中的子模型
    #     'model_path': r"D:\work\PointClassifier\new_logs\pf_chamfer\pf_tnet_noreg\checkpoints\best_pointfield.t7"  # 权重路径
    # }
    # {  # 老版本 tnet adam
    #     'type': 'PF_Tnet',  # 数据经过的pointfield模型 'PF_1layer', 'PF_3layer', 'PF_Tnet'
    #     'sub_model': False,  # pf是否是权重文件中的子模型
    #     'model_path': r"D:\work\PointClassifier\new_logs\pf\pf_tnet\checkpoints\best_pointfield.t7"  # 权重路径
    # },
    # {  # 与dgcnn一同训练的 tnet模型
    #     'type': 'PF_Tnet',  # 数据经过的pointfield模型 'PF_1layer', 'PF_3layer', 'PF_Tnet'
    #     'sub_model': True,  # pf是否是权重文件中的子模型
    #     'model_path': r"D:\work\PointClassifier\new_logs\dgcnn\tnet_sgd\checkpoints\best_model.t7"  # 权重路径
    # },
    # {  # 一层pf
    #     'type': 'PF_1layer',
    #     'sub_model': False ,
    #     'model_path': r"D:\work\PointClassifier\new_logs\pf_chamfer\pf_1layer_sgd\checkpoints\best_pointfield.t7"
    # },
    # {  # 三层pf
    #     'type': 'PF_3layer',
    #     'sub_model': False,
    #     'model_path': r"D:\work\PointClassifier\new_logs\pf_chamfer\pf_3layer_sgd\checkpoints\best_pointfield.t7"  # 权重路径
    # },
    # {  # tnet sgd
    #     'type': 'PF_Tnet',
    #     'sub_model': False,
    #     'model_path': r"D:\work\PointClassifier\new_logs\pf_chamfer\pf_tnet_sgd\checkpoints\best_pointfield.t7"  # 权重路径
    #     # model_path = r'D:\work\PointClassifier\new_logs\pf\pf_tnet\checkpoints\best_pointfield.t7'
    # },
    # {  # 无
    #     'type': 'None',
    #     'sub_model': False,
    #     'model_path': r""  # 权重路径
    # }
]

def main(dataset, save, savedir, show, type, sub_model, model_path):
    # pointfield
    chamfer_ana = ChamferAnalysis()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chamfer_ana.load_model(type, model_path, device, sub_model)  # 加载 pointfield

    # 数据集
    train_transforms = transforms.Compose([
        shuffle_pointcloud
    ])
    train = ModelNet40(partition=dataset, num_points=1024, transform=train_transforms)
    dataLoader = torch.utils.data.DataLoader(train, num_workers=1, batch_size=64, shuffle=True, drop_last=True)

    # 计算 chamfer distance 并统计
    for data, label in tqdm(dataLoader):
        data = data.to(device)
        if chamfer_ana.model is not None:
            data = chamfer_ana.model(data, label, require_loss=False)
        label = label.numpy()
        dist_mat = pdist_pc(data).detach().cpu().numpy()  # 计算 batch 的 cd 矩阵
        for i, row in enumerate(label):
            for j, col in enumerate(label):
                if i == j:
                    continue  # 跳过自己和自己的距离
                chamfer_ana.update(row, col, dist_mat[i, j])  # 更新cd矩阵的平均值和方差
    # 保存结果
    chamfer_ana.save()
    chamfer_ana.draw_heatmap(filename(dataset, type, sub_model, savedir=savedir), save=save, show=show)


if __name__ == '__main__':
    savedir = increment_path(BASE_DIR / savedir_name, exist_ok=exist_ok)  # 创建本次使用的保存目录
    for ds in dataset:
        for setting in settings:
            main(ds, save, savedir, show, **setting)
