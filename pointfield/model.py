import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointfield.losses import get_triplet_loss
from pointfield.visualize_utils import draw_grid, interval_concatenate
import cv2
import numpy as np
from pointfield.tnet import STN3d

def load_sub_state_dict(sub_model, state_dict, sub_name='classifier'):
    """ 加载模型参数到模型成员中 """
    sub_dict = {}
    pattern = '^.*' + sub_name + '\.'  # 匹配 '...pointfield.'
    pattern = re.compile(pattern, re.I)
    for key, val in state_dict.items():
        if pattern.match(key):
            new_key = pattern.sub('', key)  # 去掉前缀pointfield
            sub_dict[new_key] = val
    sub_model.load_state_dict(sub_dict)


class PointField(nn.Module):
    """ PointField层 """
    def __init__(self, dim, initial='normal'):
        super().__init__()
        grid = torch.zeros([3, dim, dim, dim], dtype=torch.float32, device='cuda')
        if initial == 'normal':
            grid.normal_(0, 0.1)
        self.grid = nn.parameter.Parameter(grid)
        self.dim = dim

    # label: B, int
    def forward(self, x):
        grid = self.grid[None].expand(x.shape[0], -1, -1, -1, -1)  # B x 3 x D x D x D
        if x.shape[-1] == 6:
            points = x[..., :3]
        else:
            points = x
        idx = points[:, :, None, None]   # to fill in [-1, 1], B x P x 1 x 1 x 3
        flow = F.grid_sample(grid, idx, align_corners=False)[:, :, :, 0, 0].transpose(1, 2) # B x 3 x P -> B x P x 3
        if x.shape[-1] == 6:
            return torch.cat([points + flow, x[..., 3:]], dim=-1)
        else:
            return x + flow

    def draw_grid(self):
        return draw_grid(self.grid)


class PointFieldTnet(nn.Module):
    """ PointField层 """
    def __init__(self, dim, initial='normal'):
        super().__init__()
        self.stn = STN3d(channel=3)
        grid = torch.zeros([3, dim, dim, dim], dtype=torch.float32, device='cuda')
        if initial == 'normal':
            grid.normal_(0, 0.1)
        self.grid = nn.parameter.Parameter(grid)
        self.dim = dim

    # label: B, int
    def forward(self, x):
        grid = self.grid[None].expand(x.shape[0], -1, -1, -1, -1)  # B x 3 x D x D x D
        if x.shape[-1] == 6:
            points = x[..., :3]
        else:
            points = x
        points = points.transpose(2, 1)  # B x 3or6 x 1024
        rot, trans = self.stn(points)  # transform
        points = points.transpose(2, 1)  # B x 1024 x 3or6
        idx = torch.bmm(points, rot) / 2 + trans.unsqueeze(1)
        idx = idx[:, :, None, None]   # to fill in [-1, 1], B x P x 1 x 1 x 3
        flow = F.grid_sample(grid, idx, align_corners=False)[:, :, :, 0, 0].transpose(1, 2) # B x 3 x P -> B x P x 3
        if x.shape[-1] == 6:
            return torch.cat([points + flow, x[..., 3:]], dim=-1)
        else:
            return x + flow

    def draw_grid(self):
        return draw_grid(self.grid)


class PFWithLoss(nn.Module):
    """ PointField 包括loss的模型基类
    输入为点云数据, 以及label, 输出偏移点云以及用于训练的 reg_loss 和 tri_loss
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, label, require_loss=True):
        """ return x_out, reg_loss, tri_loss """
        raise NotImplementedError

    @classmethod
    def creat_pf(cls, config):
        raise NotImplementedError

    def draw_grid(self):
        """ 训练时, 若track_grid参数为True, 则调用该函数 """
        pass


class PF_1layer(PFWithLoss):
    """ 单层PointField模型 """
    def __init__(self, dim=64, initial='normal', tri_criterion=None):
        super().__init__()
        self.pf = PointField(dim, initial)
        self.tri_criterion = tri_criterion

    def forward(self, x, label, require_loss=True):
        x = self.pf(x)

        if require_loss:  # calc reg_loss and tri_loss
            reg_loss = self.pf.grid.abs().mean()
            tri_loss, _ = self.tri_criterion(x, label)
            return x, tri_loss, reg_loss
        return x

    def draw_grid(self):
        return self.pf.draw_grid()

    @classmethod
    def creat_pf(cls, config):
        tri_criterion = get_triplet_loss(config["loss"])
        return cls(config['dims'], config['initial'], tri_criterion)


class PF_Nlayer(PFWithLoss):
    """ n层PointField模型 """
    def __init__(self, dims: list, initial='normal', tri_criterion=None):
        super().__init__()
        self.pf_layers = nn.ModuleList()
        for dim in dims: # n pointfield layers
            self.pf_layers.append(PointField(dim, initial))
        self.tri_criterion = tri_criterion
        self._dims = dims

    def forward(self, x, label, require_loss=True):
        for pf in self.pf_layers:
            x = pf(x)

        if require_loss:  # calc reg_loss and tri_loss
            reg_loss = torch.tensor(0, dtype=float)
            for pf in self.pf_layers:  # 计算 reg_loss
                reg_loss += pf.grid.abs().mean().cpu()
            tri_loss, _ = self.tri_criterion(x, label)  # 计算 tri_loss
            return x, tri_loss, reg_loss
        return x

    def draw_grid(self):
        """ 可视化n个pointfield的 grid 于一张图片 """
        grid_imgs = []
        img_sizes = []
        for pf in self.pf_layers:  # 绘制每层的grid
            img = pf.draw_grid()
            img_sizes.append(img.shape[0])
            grid_imgs.append(img)

        max_size = max(img_sizes)
        for i, pf_img in enumerate(grid_imgs):  # 将每张图填充成一样大小
            if pf_img.shape[0] < max_size:
                pad_size = max_size - pf_img.shape[0]
                grid_imgs[i] = np.pad(pf_img, ((0, pad_size)(0, pad_size), (0, 0)))

        for pf_img in grid_imgs: # 连接图片
            grid_img = interval_concatenate(grid_imgs, axis=1, interval=10, fill_value=0)

        return grid_img

    @classmethod
    def creat_pf(cls, config):
        tri_criterion = get_triplet_loss(config["loss"])
        return cls(config['dims'], config['initial'], tri_criterion)


class PF_Tnet(PFWithLoss):
    """ 单层PointField模型 """
    def __init__(self, dim=64, initial='normal', tri_criterion=None):
        super().__init__()
        self.pf = PointFieldTnet(dim, initial)
        self.tri_criterion = tri_criterion

    def forward(self, x, label, require_loss=True):
        x = self.pf(x)

        if require_loss:  # calc reg_loss and tri_loss
            reg_loss = self.pf.grid.abs().mean()
            tri_loss, _ = self.tri_criterion(x, label)
            return x, tri_loss, reg_loss
        return x

    def draw_grid(self):
        return self.pf.draw_grid()

    @classmethod
    def creat_pf(cls, config):
        tri_criterion = get_triplet_loss(config["loss"])
        return cls(config['dims'], config['initial'], tri_criterion)


class PFWithModel(nn.Module):
    """ 将点云分类网络与Pointfield合并为一个网络
    输入 x, label,
    输出 logits或者logsoftmax(logits), loss
    """
    def __init__(self, classifier, criterion, pf_config):
        """初始化

        Args:
            classifier : 点云分类器
            criterion: 分类loss函数
            pf_config (dict): 配置
        """
        super().__init__()
        self.classifier = classifier
        self.criterion = criterion
        self.weights = pf_config['weights']
        # 创建 pointfield
        if pf_config["type"]=="PF_1layer":
            self.pointfield = PF_1layer.creat_pf(pf_config)
        if pf_config["type"]=="PF_Nlayer":
            self.pointfield = PF_Nlayer.creat_pf(pf_config)
        if pf_config["type"]=="PF_Tnet":
            self.pointfield = PF_Tnet.creat_pf(pf_config)
        # 加载预训练
        if pf_config["pf_path"] is not None and not pf_config["resume"]:
            print('Use pre-trained pointfield: ' + pf_config["pf_path"])
            self.pointfield.load_state_dict(torch.load(pf_config["pf_path"])['model_state_dict'])
        if pf_config["model_path"] is not None and not pf_config["resume"]:
            print('Use pre-trained classifier: ' + pf_config["model_path"])
            load_sub_state_dict(self.classifier,
                torch.load(pf_config["model_path"])['model_state_dict'],
                sub_name='classifier')
        # freeze
        self._use_pf = pf_config["use_pf"]
        self._freeze_pf = pf_config["freeze_pf"]
        self._freeze_model = pf_config["freeze_model"]
        if self._freeze_pf:
            for param in self.pointfield.parameters():
                param.requires_grad = False
        if self._freeze_model:
            for param in self.classifier.parameters():
                param.requires_grad = False

    def forward(self, x, label):
        if self.training: # 训练阶段, 返回logits 以及 loss
            tri_loss, reg_loss = torch.tensor(0), torch.tensor(0)
            if self._use_pf:  # 使用 pf
                if self._freeze_pf:  # 不训练 pf
                    x = self.pointfield(x, label, require_loss=False)
                else:
                    x, tri_loss, reg_loss = self.pointfield(x, label, require_loss=True)

            logits = self.classifier(x)
            cls_loss = self.criterion(logits, label)
            return logits, self.weights[0]*tri_loss, self.weights[1]*reg_loss, self.weights[2]*cls_loss

        else:
            if self._use_pf:
                x = self.pointfield(x, label, require_loss=False)
            logits = self.classifier(x)
            return logits








