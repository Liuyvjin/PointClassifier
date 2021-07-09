#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""


import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

def farthest_point_sample(point, npoint):
    """
    Input:
        point: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10  # distances to sampled pointcloud
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def normalize_pointcloud(pointcloud):
    xyz = pointcloud[:,:3]
    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    m = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
    pointcloud[:,:3] = xyz / m
    return pointcloud

def dropout_pointcloud(pointcloud, max_dropout_ratio=0.875):
    dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
    drop_idx = np.where(np.random.random((pointcloud.shape[0]))<=dropout_ratio)[0]
    if len(drop_idx)>0:
        pointcloud[drop_idx,:] = pointcloud[0,:] # set to the first point
    return pointcloud

def scale_pointcloud(pointcloud, scale_low=0.8, scale_high=1.25):
    pointcloud[:,:3] = pointcloud[:,:3] * np.random.uniform(scale_low, scale_high)
    return pointcloud

def shift_pointcloud(pointcloud, shift_range=0.1):
    pointcloud[:,:3] = pointcloud[:,:3] + np.random.uniform(-shift_range, shift_range, (1,3))
    return pointcloud

def translate_pointcloud(pointcloud):
    xyz = pointcloud[:,:3]
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    pointcloud[:,:3] = np.add(np.multiply(xyz, xyz1), xyz2).astype('float32')
    return pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud

def shuffle_pointcloud(pointcloud):
    np.random.shuffle(pointcloud)
    return pointcloud

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR =  BASE_DIR  #os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition, normal_channel):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = BASE_DIR #os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    all_normal = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name, mode='r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        if normal_channel:
            normal = f['normal'][:].astype('float32')
            all_normal.append(normal)
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)

    if normal_channel:
        all_normal = np.concatenate(all_normal, axis=0)
        all_data = np.concatenate([all_data, all_normal], axis=2)
    return all_data, all_label


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train', model=None, uniform=False, normal_channel=False, transform=None):
        self.data, self.label = load_data(partition, normal_channel)
        self.num_points = num_points
        self.partition = partition
        self.uniform = uniform
        self.transform = transform
        self.model = model

    def __getitem__(self, item):
        label = self.label[item]
        pointcloud = self.data[item]

        # sample
        if self.uniform:
            pointcloud = farthest_point_sample(pointcloud, self.num_points)
        else:
            pointcloud = pointcloud[:self.num_points]

        # model behavior
        if self.model in ['pointnet', 'pointnet2']:
            pointcloud = normalize_pointcloud(pointcloud)

        # transform
        if self.partition == 'train' and self.transform:
            pointcloud = self.transform(pointcloud)

        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


dgcnn_transforms = transforms.Compose([ translate_pointcloud,
                                        shuffle_pointcloud ])


if __name__ == '__main__':
    import torch
    train_transforms = transforms.Compose([
        shuffle_pointcloud
    ])
    train = ModelNet40(1024, transform=train_transforms)
    test = ModelNet40(1024, 'test')

    DataLoader = torch.utils.data.DataLoader(train, batch_size=12, shuffle=True)
    for data, label in DataLoader:
        print(data.shape)
        print(label.shape)
