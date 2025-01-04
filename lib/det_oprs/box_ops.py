# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
import torchvision.ops
from torchvision.ops.boxes import box_area
from copy import deepcopy
import math
def filter_boxes_opr(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = (ws >= min_size) * (hs >= min_size)
    return keep

def clip_boxes_opr(boxes, im_info):
    """ Clip the boxes into the image region."""
    w = im_info[1] - 1
    h = im_info[0] - 1
    boxes[:, 0::4] = boxes[:, 0::4].clamp(min=0, max=w)
    boxes[:, 1::4] = boxes[:, 1::4].clamp(min=0, max=h)
    boxes[:, 2::4] = boxes[:, 2::4].clamp(min=0, max=w)
    boxes[:, 3::4] = boxes[:, 3::4].clamp(min=0, max=h)
    return boxes
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)

def loc2bbox(src_bbox, loc) :
    if src_bbox.size()[0] == 0 :
        return torch.zeros((0, 4), dtype = loc.dtype)

    src_width = torch.unsqueeze(src_bbox[:, 2] - src_bbox[:, 0], -1)
    src_height = torch.unsqueeze(src_bbox[:, 3] - src_bbox[:, 1], -1)
    src_ctr_x = torch.unsqueeze(src_bbox[:, 0], -1) + 0.5 * src_width
    src_ctr_y = torch.unsqueeze(src_bbox[:, 1], -1) + 0.5 * src_height

    dx = loc[:, 0 : :4]
    dy = loc[:, 1 : :4]
    dw = loc[:, 2 : :4]
    dh = loc[:, 3 : :4]

    ctr_x = dx * src_width + src_ctr_x
    ctr_y = dy * src_height + src_ctr_y
    w = torch.exp(dw) * src_width
    h = torch.exp(dh) * src_height

    dst_bbox = torch.zeros_like(loc)
    dst_bbox[:, 0 : :4] = ctr_x - 0.5 * w
    dst_bbox[:, 1 : :4] = ctr_y - 0.5 * h
    dst_bbox[:, 2 : :4] = ctr_x + 0.5 * w
    dst_bbox[:, 3 : :4] = ctr_y + 0.5 * h

    return dst_bbox

def bbox2loc(src_bbox, dst_bbox) :
    width = src_bbox[:, 2] - src_bbox[:, 0]
    height = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_x = src_bbox[:, 0] + 0.5 * width
    ctr_y = src_bbox[:, 1] + 0.5 * height

    base_width = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_height = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_x = dst_bbox[:, 0] + 0.5 * base_width
    base_ctr_y = dst_bbox[:, 1] + 0.5 * base_height

    eps = torch.tensor(torch.finfo(torch.float16).tiny)
    width = torch.maximum(width, eps)
    height = torch.maximum(height, eps)

    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height
    dw = torch.log(base_width / width)
    dh = torch.log(base_height / height)

    loc = torch.stack((dx, dy, dw, dh), dim = 1)
    return loc

def find_argmax(t, iou_threshlod = 0.5) :
    h, w = t.shape
    l = torch.zeros([w,], dtype = torch.int8)
    iou = []
    for i in range(w) :
        argmax = torch.argmax(t[:, i])
        current_iou = t[argmax, i]
        iou.append(deepcopy(t[argmax, i]))
        if current_iou >= iou_threshlod :
            t[argmax] = -1
            l[i] = 1
    return l, torch.tensor(iou)

def box_overlap_ignore_opr(box, gt, ignore_label=-1):
    assert box.ndim == 2
    assert gt.ndim == 2
    assert gt.shape[-1] == 4
    area_box = (box[:, 2] - box[:, 0] + 1) * (box[:, 3] - box[:, 1] + 1)
    area_gt = (gt[:, 2] - gt[:, 0] + 1) * (gt[:, 3] - gt[:, 1] + 1)
    width_height = torch.min(box[:, None, 2:], gt[:, 2:4]) - torch.max(
        box[:, None, :2], gt[:, :2])  # [N,M,2]
    width_height.clamp_(min=0)  # [N,M,2]
    inter = width_height.prod(dim=2)  # [N,M]
    del width_height
    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (area_box[:, None] + area_gt - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device))
    ioa = torch.where(
        inter > 0,
        inter / (area_box[:, None]),
        torch.zeros(1, dtype=inter.dtype, device=inter.device))
    # gt_ignore_mask = gt[:, 4].eq(ignore_label).repeat(box.shape[0], 1)

    gt_ignore_mask = torch.zeros_like(iou).long()
    iou *= ~gt_ignore_mask
    ioa *= gt_ignore_mask
    return iou, ioa
def label_box(pred_boxes, gt_boxes, iou_threshold = 0.5):

    if not torch.is_tensor(pred_boxes):
        pred_boxes = torch.tensor(pred_boxes)

    if not torch.is_tensor(gt_boxes):
        gt_boxes = torch.tensor(gt_boxes)

    ious = torchvision.ops.box_iou(gt_boxes, pred_boxes)
    box_label, iou = find_argmax(ious, iou_threshold)
    return box_label, iou


def box_ioa(boxes1, boxes2):
    area1 = box_area(boxes1)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    return inter / area1[:, None]