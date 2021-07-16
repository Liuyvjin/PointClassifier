import torch
from torch import functional
import os.path as osp
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import numpy as np


def show_grid(grid, channel=None, save=False, img_name=None):
    '''
    input grid (dim, dim, dim, 3), numpy
    '''
    if isinstance(grid, torch.Tensor):
        grid = grid.cpu().detach().permute(1,2,3,0).numpy()
    dim, _, _, _ = grid.shape
    n = int(np.sqrt(dim))
    # normalize
    max_val = np.array([np.max(grid[..., i]) for i in range(3)])
    min_val = np.array([np.min(grid[..., i]) for i in range(3)])
    range_val = max_val - min_val
    grid = np.multiply((grid - min_val) / range_val, 255).astype('uint8')

    # concatnate
    arrays = []
    for i in range(n):
        line_array = interval_concatenate(grid[i*n : (i+1)*n], axis=1, interval=5)
        arrays.append(line_array)
    image = interval_concatenate(arrays, axis=0, interval=5)

    # type
    image = image.astype('uint8')
    image = to_pil_image(image)
    if channel in ('R', 'G', 'B'):
        image = image.getchannel(channel)
    if save:
        image.save(img_name)
    else:
        image.show()


def interval_concatenate(arrays, axis=0, interval=0, fill_value=255):
    # arrays: list of arrays
    # axis: stack on axis
    n = len(arrays)
    if interval != 0:
        fill_shape = list(arrays[0].shape)
        fill_shape[axis] = interval
        fill_array = np.full(fill_shape, fill_value)
        arrays = [arrays[i//2] if i%2==0 else fill_array for i in range(2*n-1)]
    return np.concatenate(arrays, axis)