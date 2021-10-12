import torch
# from torchvision.transforms.functional import to_pil_image
import cv2
import numpy as np


def draw_grid(grid):
    """可视化 pointfeild grid

    Args:
        grid (numpy): PointFieldLayer的grid (dim, dim, dim, 3)
    """
    if isinstance(grid, torch.Tensor):
        grid = grid.cpu().detach().permute(1,2,3,0).numpy()
    dim, _, _, _ = grid.shape
    n = int(np.sqrt(dim))  # n行 n列图片
    # normalize
    max_val = np.max(grid)
    min_val = np.min(grid)
    range_val = max_val - min_val
    grid = np.multiply((grid - min_val) / range_val, 255).astype('uint8')

    # concatnate
    arrays = []
    for i in range(n):
        line_array = interval_concatenate(grid[i*n : (i+1)*n], axis=1, interval=5)  # 连接一行的图片
        arrays.append(line_array)
    image = interval_concatenate(arrays, axis=0, interval=5)  # 纵向连接
    return image.astype('uint8')


def interval_concatenate(arrays, axis=0, interval=0, fill_value=255):
    """
    arrays: list of arrays
    axis: stack on axis
    """
    n = len(arrays)
    if interval != 0:
        fill_shape = list(arrays[0].shape)
        fill_shape[axis] = interval
        fill_array = np.full(fill_shape, fill_value)
        arrays = [arrays[i//2] if i%2==0 else fill_array for i in range(2*n-1)]
    return np.concatenate(arrays, axis)


if __name__ == '__main__':
    img1 = np.random.randint(0, 255, (65,65,65,3)).astype('uint8')
    # img2 = np.random.randint(0, 255, (32,32,32,3)).astype(uint8)
    img = draw_grid(img1)
    img = cv2.resize(img, (300,200))
    ii = [img, ]
    for x in ii:
        x = np.pad(x, ((0, 25), (0,10), (0,0)))
    print(ii[0].shape)

    cv2.imshow('test', img)
    cv2.waitKey(0)