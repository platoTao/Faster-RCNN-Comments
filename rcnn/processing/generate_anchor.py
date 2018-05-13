"""
Generate base anchors on index 0
"""

import numpy as np


def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """
    # (左上坐标,右下坐标)，之所以是16，是因为特征图尺寸是图片的1/16，所以这块原始区域对应特征图上一个点
    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    # 宽高比扩展:纵框,平框,横框
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    # 在base anchor大小的基础上针对大小扩展: x8, x16, x32
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in xrange(ratio_anchors.shape[0])])
    return anchors


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    # 转换成w,h,中心坐标
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    # base anchor是一个正方形,假设边长为n, new w = n/(√radio), new h = n*√radio,新的边长具有如下特点:面积大体不变(忽略上下round的损失),w/h = radio,也就说这样计算完在面积大体不变的情况下:实现宽高按照raio设定的比例走,有点像拉长和压扁

    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    # 转成坐标形式,_whctrs的逆操作
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

# 按照面积比例扩展,实际是scales元素的平方扩展
def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors
