import numpy as np


def unique_boxes(boxes, scale=1.0):
    """ return indices of unique boxes """
    # 如果x1,y1,x2,y2相等的话，和v点乘的结果也应该相等
    v = np.array([1, 1e3, 1e6, 1e9])
    hashes = np.round(boxes * scale).dot(v)
    # 去掉重复元素
    _, index = np.unique(hashes, return_index=True)
    return np.sort(index)

# 过滤掉较小的boxes
def filter_small_boxes(boxes, min_size):
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = np.where((w >= min_size) & (h > min_size))[0]
    return keep
