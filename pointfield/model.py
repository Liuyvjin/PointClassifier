import torch
import torch.nn as nn
import torch.nn.functional as F
from pointfield.losses import OnlineTripletLoss
from pointfield.loss_utils import HardestNegativeTripletSelector, pdist_pc, chamfer_pc
from pointfield.visualize_utils import draw_grid, interval_concatenate
import cv2
import numpy as np

class PointFieldLayer(nn.Module):
    """ PointField层 """
    def __init__(self, dim):
        super().__init__()
        grid = torch.zeros([3, dim, dim, dim], dtype=torch.float32, device='cuda')
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
        idx = points[:, :, None, None] / 1.5  # to fill in [-1, 1], B x P x 1 x 1 x 3
        flow = F.grid_sample(grid, idx, align_corners=False)[:, :, :, 0, 0].transpose(1, 2) # B x 3 x P -> B x P x 3
        if x.shape[-1] == 6:
            return torch.cat([points + flow, x[..., 3:]], dim=-1)
        else:
            return x + flow

    def draw_grid(self):
        return draw_grid(self.grid)


""" 默认TripletLoss """
selector = HardestNegativeTripletSelector(0.3, cpu=False, dist_func=pdist_pc)
get_loss = OnlineTripletLoss(   margin = 0.3,
                                triplet_selector = selector,
                                dist_func = chamfer_pc)


class PointField(nn.Module):
    """ PointField 模型基类
    输入为点云数据, 以及label, 输出偏移点云以及用于训练的 reg_loss 和 tri_loss
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, label, require_loss=True):
        """ return x_out, reg_loss, tri_loss """
        raise NotImplementedError

    def draw_grid(self):
        """ 训练时, 若track_grid参数为True, 则调用该函数 """
        pass

class PF_1layer(PointField):
    """ 单层PointField模型 """
    def __init__(self, dim=64, tri_criterion=get_loss):
        super().__init__()
        self.pf = PointFieldLayer(dim)
        self.tri_criterion = tri_criterion

    def forward(self, x, label, require_loss=True):
        x = self.pointfield(x)

        if require_loss:  # calc reg_loss and tri_loss
            reg_loss = self.pf.grid.abs().mean()
            tri_loss, _ = self.tri_criterion(x, label)
            return x, tri_loss, reg_loss
        return x

    def draw_grid(self):
        return self.pf.draw_grid()


class PF_Nlayer(PointField):
    """ n层PointField模型 """
    def __init__(self, dims: list, tri_criterion=get_loss):
        super().__init__()
        self.pf_layers = nn.ModuleList()
        for dim in dims: # n pointfield layers
            self.pf_layers.append(PointFieldLayer(dim))
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



class PF_Model(nn.Module):
    """ 将点云分类网络与Pointfield合并为一个网络
    输入 x, label,
    输出 logits或者logsoftmax(logits), loss
    """
    def __init__(self, classifier, pointfield, criterion,
                use_pointfield=True, train_pointfield=True, pointfield_path=None):
        """初始化

        Args:
            classifier : 点云分类器
            pointfield (PointField): PointField实例
            criterion: 分类loss函数
            use_pointfield (bool, optional): 是否在训练和测试时使用PointField. Defaults to True.
            train_pointfield (bool, optional): 是否同时训练PointField和classfier, 在_use_pf为真时生效. Defaults to True.
            pointfield_path (str, optional): 预训练PointField模型路径. Defaults to None.
        """
        super().__init__()
        self.classifier = classifier
        self.criterion = criterion
        self.pointfield = pointfield
        if pointfield_path is not None:  # 加载预训练 PointField
            self.pointfield.load_state_dict(torch.load(pointfield_path)['model_state_dict'])
        self._use_pf = use_pointfield
        self._train_pf = train_pointfield

    def forward(self, x, label):
        if self.training: # 训练阶段, 返回logits 以及 loss
            tri_loss, reg_loss = torch.tensor(0), torch.tensor(0)
            if self._use_pf:  # 使用 pf
                if self._train_pf:  # 训练 pf
                    x, tri_loss, reg_loss = self.pointfield(x, label, required_loss=True)
                else:
                    x = self.pointfield(x, label, required_loss=False)
                    x = x.detach()

            logits = self.classifier(x)
            cls_loss = self.criterion(logits, label)
            return logits, tri_loss, reg_loss, cls_loss

        else:
            if self._use_pf:
                x = self.pointfield(x, label, required_loss=False)
            logits = self.classifier(x)
            return logits



class CombinedModel(nn.Module):
    def __init__(self, classifier, pointfield_dim=64, tri_criterion=None, use_pointfield=True, detach=False, pointfield_path=None):
        """初始化

        Args:
            classifier : 点云分类网络
            pointfield_dim (int, optional): pointfield维数. Defaults to 64.
            tri_criterion ([type], optional): triplet loss. Defaults to None.
            use_pointfield (bool, optional): 是否使用pointfield. Defaults to True.
            detach (bool, optional): 是否将pointfield分开训练. Defaults to False.
            pointfield_path ([type], optional): 预训练 pointfield路径. Defaults to None.
        """
        super().__init__()
        self.use_pointfield = use_pointfield
        self.pointfield = PointField(pointfield_dim)
        self.classifier = classifier
        if tri_criterion == None:
            self.tri_criterion = get_loss
        else:
            self.tri_criterion = tri_criterion
        self.reg_loss = torch.tensor(0)
        self.tri_loss = torch.tensor(0)
        self.detach = detach
        if pointfield_path is not None:
            self.pointfield.load_state_dict(torch.load(pointfield_path)['model_state_dict'])

    def forward(self, x, label=None):
        if self.use_pointfield:
            x = self.pointfield(x)
            if self.detach:
                x = x.detach()
            elif self.training:  # update reg_loss and tri_loss
                self.reg_loss = self.pointfield.grid.abs().mean()
                self.tri_loss, _ = self.tri_criterion(x, label)

        return self.classifier(x)






