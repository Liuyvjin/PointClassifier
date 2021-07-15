import torch
import torch.nn as nn
import torch.nn.functional as F
from pointfield.losses import OnlineTripletLoss
from pointfield.loss_utils import HardestNegativeTripletSelector, pdist_pc, chamfer_pc

# chamfer distance: sum_i min_j(x[i] - y[j]) + sum_j min_i(x[i] - y[j])
# 83.2% 93%
class PointField(nn.Module):
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

class CombinedModel(nn.Module):
    def __init__(self, classifier, point_field_dim=64, tri_criterion=None, use_pointfield = True):
        super().__init__()
        self.use_pointfield = use_pointfield
        self.pointfield = PointField(point_field_dim)
        self.classifier = classifier

        if tri_criterion == None:
            selector = HardestNegativeTripletSelector(0.3, cpu=False, dist_func=pdist_pc)
            tri_criterion = OnlineTripletLoss(  margin = 0.3,
                                                triplet_selector = selector,
                                                dist_func = chamfer_pc)
        self.tri_criterion = tri_criterion
        self.reg_loss = torch.tensor(0)
        self.tri_loss = torch.tensor(0)

    def forward(self, x, label=None):
        if self.use_pointfield:
            x = self.pointfield(x)
            if self.training:  # update reg_loss and tri_loss
                self.reg_loss = self.pointfield.grid.abs().mean()
                self.tri_loss, _ = self.tri_criterion(x, label)
        return self.classifier(x)



