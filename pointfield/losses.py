import torch
import torch.nn as nn
import torch.nn.functional as F
from pointfield.loss_utils import (HardestNegativeTripletSelector, RandomNegativeTripletSelector,
    SemihardNegativeTripletSelector, chamfer_pc, pdist_pc)

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector, dist_func):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector
        self.dist_func = dist_func

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()

        positive_loss = self.dist_func(embeddings[positive_pairs[:, 0]], embeddings[positive_pairs[:, 1]])
        negative_loss = -self.dist_func(embeddings[negative_pairs[:, 0]], embeddings[negative_pairs[:, 1]])
        # loss = torch.cat([positive_loss, negative_loss], dim=0)
        return positive_loss, negative_loss


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector, dist_func):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector
        self.dist_func = dist_func

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = self.dist_func(embeddings[triplets[:, 0]], embeddings[triplets[:, 1]])  # .pow(.5)
        an_distances = self.dist_func(embeddings[triplets[:, 0]], embeddings[triplets[:, 2]])  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)


def get_triplet_loss(config):
    margin = config['margin']
    if config['selector'] == 'hardest_negtive':
        Selector = HardestNegativeTripletSelector(margin, cpu=False, dist_func=pdist_pc)
    elif config['selector'] == 'random_negtive':
        Selector = RandomNegativeTripletSelector(margin, cpu=False, dist_func=pdist_pc)
    elif config['selector'] == 'semihard_negtive':
        Selector = SemihardNegativeTripletSelector(margin, cpu=False, dist_func=pdist_pc)
    return OnlineTripletLoss(margin=margin, triplet_selector=Selector, dist_func=chamfer_pc)