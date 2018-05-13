#非极大值抑制
import numpy as np
from ..cython.cpu_nms import cpu_nms
try:
    from ..cython.gpu_nms import gpu_nms
except ImportError:
    gpu_nms = None


def py_nms_wrapper(thresh):
    def _nms(dets):
        return nms(dets, thresh)
    return _nms


def cpu_nms_wrapper(thresh):
    def _nms(dets):
        return cpu_nms(dets, thresh)
    return _nms


def gpu_nms_wrapper(thresh, device_id):
    def _nms(dets):
        return gpu_nms(dets, thresh, device_id)
    if gpu_nms is not None:
        return _nms
    else:
        return cpu_nms_wrapper(thresh)


def nms(dets, thresh):
    """
    选择IoU小于阈值中的具有最大置信度即得分的boxs
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    排除 IoU 大于等于 阈值
    rule out overlap >= thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    # 计算面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 按得分从大到小返回对应的索引
    order = scores.argsort()[::-1]

    keep = []
    # 依次选取最高得分的边界框，排除与它重叠超过阈值的边界框
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算IoU,两个的交集除以两个的并集
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 与目前最高得分的边界框重叠超过阈值的被丢弃
        inds = np.where(ovr <= thresh)[0]
        # 这里为什么所有索引 +1 ？
        order = order[inds + 1]

    return keep
