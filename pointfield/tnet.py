import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


def rot_gram_schmidt(vec1, vec2):
    """根据Gram-Schmidt正交化生成旋转矩阵

    Args:
        vec1 [tensor]: B x 3
        vec2 [tensor]: B x 3
    return:
        matrix: B x 3 x 3
    """
    vec1 = F.normalize(vec1, dim=1)  # 标准化向量1
    vec2 = vec2 - torch.mul(vec1, vec2).sum(dim=1, keepdim=True) * vec1  # 正交化
    vec2 = F.normalize(vec2)  # 单位化
    vec3 = torch.cross(vec1, vec2, dim=1)  # 叉乘, 单位向量
    return torch.stack((vec1, vec2, vec3), dim=1)


class STN3d(nn.Module):
    """为每个点云生成一个 3x3 的rotate 和 3x1的translate
    """
    def __init__(self, channel=3):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 6)
        self.fc4 = nn.Linear(256, 3)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(3)

    def forward(self, x):
        # batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))  # B x 1024 x n
        x = torch.max(x, 2, keepdim=True)[0]  # maxpool, B x 1024 x 1
        x = x.view(-1, 1024)  # B x 1024

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        rotate = self.fc3(x)  # B x 6 : 包括转轴三维, 旋转角度一维rad
        rotate = rot_gram_schmidt(rotate[:, :3], rotate[:, 3:])  # 施密特正交化构造三维旋转矩阵
        translate = torch.sigmoid(self.fc4(x)) - 0.5  # B x 3 , 使用sigmoid归一化 G~-0.5,0.5
        return rotate, translate

        # iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
        #     batchsize, 1)
        # if rotate.is_cuda:
        #     iden = iden.cuda()
        # rotate = rotate + iden
        # rotate = rotate.view(-1, 3, 3)  # B x 3 x 3


if __name__ == '__main__':
    a = torch.tensor([[1,2,3], [2,2,2], [0,1,0]], dtype=float)
    b = torch.tensor([[1,2,1], [2,2,2], [0,0,0]], dtype=float)
    c = rot_gram_schmidt(a,b)
    print(c[0].mm(c[0].T))