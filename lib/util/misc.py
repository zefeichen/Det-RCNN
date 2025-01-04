# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import datetime
import os
import pickle
import subprocess
import time
from argparse import Namespace
from collections import defaultdict, deque
from typing import List, Optional
import json
import torch
import torch.distributed as dist
import torch.nn.functional as F
# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision
from torch import Tensor
from visdom import Visdom
from PIL import Image
from torchvision.ops import nms, box_iou
from scipy.optimize import linear_sum_assignment


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{value:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):

    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, print_freq, delimiter="\t", record_file = None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.print_freq = print_freq
        self.record_file = record_file

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, epoch=None, header=None):
        i = 0
        if header is None:
            header = 'Epoch: [{}]'.format(epoch)

        world_len_iterable = get_world_size() * len(iterable)

        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(world_len_iterable))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'time: {current_time}',
                'eta: {eta}',
                'spent time: {spent_time}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'time : {current_time}',
                'eta: {eta}',
                'spent time : {spent_time}',
                '{meters}',
                'time: {time}',
                'data_time: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % self.print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                spent_seconds = iter_time.global_avg * i
                spent_string = str(datetime.timedelta(seconds=int(spent_seconds)))

                if torch.cuda.is_available():
                    info = log_msg.format(
                        i * get_world_size(), world_len_iterable, current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        eta=eta_string, spent_time = spent_string, meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB)

                else:
                    info = log_msg.format(
                        i * get_world_size(), world_len_iterable, eta=eta_string,
                        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        meters=str(self), time=str(iter_time), data=str(data_time))
                print(info)
                if self.record_file:
                    with open(self.record_file, 'a+') as file :
                        print(info, file = file)

            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, _, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)



def faster_find_argmax(t, iou_threshlod = 0.5) :
    t = t.cpu()
    matrix = t > iou_threshlod
    gt_m_idx, dt_m_idx = linear_sum_assignment(matrix.cpu(), True)
    gt_m_idx, dt_m_idx = torch.from_numpy(gt_m_idx), torch.from_numpy(dt_m_idx)
    overlaps = t[gt_m_idx, dt_m_idx]
    keep = overlaps >= iou_threshlod
    gt_m_idx = gt_m_idx[keep]
    dt_m_idx = dt_m_idx[keep]
    overlaps = overlaps[keep]
    return gt_m_idx, dt_m_idx, overlaps
#
# def body_match(outputs, outputs_head_decoder, thr = 0.01) :
#     results = []
#     for output, output_head_decoder in zip(outputs, outputs_head_decoder) :
#         scores = output['scores']
#         labels = output['labels']
#         boxes = output['boxes']
#
#         scores_head_decoder = output_head_decoder['scores']
#         labels_head_decoder = output_head_decoder['labels']
#         boxes_head_decoder = output_head_decoder['boxes']
#
#         body_mask = (labels == 0) & (scores > thr)
#         head_mask = (labels == 1) & (scores > thr) & (scores_head_decoder > 0.01)
#
#         body_boxes = boxes[body_mask]
#         body_scores = scores[body_mask]
#
#         head_boxes = boxes[head_mask]
#         head_scores = scores[head_mask]
#         body_boxes_to_head = boxes_head_decoder[head_mask]
#
#         ious = box_iou(body_boxes, body_boxes_to_head)
#         gt_m_idx, dt_m_idx, overlaps = faster_find_argmax(ious, iou_threshlod = 0.3)
#
#         match_head_boxes = head_boxes[dt_m_idx]
#         match_head_scores = head_scores[dt_m_idx]
#
#         match_body_boxes = body_boxes[gt_m_idx]
#         match_body_scores = body_scores[gt_m_idx]
#
#         unmatch_head_mask = torch.ones(len(head_boxes)).bool()
#         unmatch_head_mask[dt_m_idx] = False
#
#         match_head_boxes = torch.cat([match_head_boxes, head_boxes[unmatch_head_mask]], dim = 0)
#         match_head_scores = torch.cat([match_head_scores, head_scores[unmatch_head_mask]], dim = 0)
#
#         match_body_boxes = torch.cat([match_body_boxes, body_boxes_to_head[unmatch_head_mask]], dim = 0)
#         match_body_scores = torch.cat([match_body_scores, head_scores[unmatch_head_mask]], dim = 0)
#
#         unmatch_body_mask = torch.ones((len(body_boxes))).bool()
#         unmatch_body_mask[gt_m_idx] = False
#
#         unmatch_body_boxes = body_boxes[unmatch_body_mask]
#         unmatch_body_scores = body_scores[unmatch_body_mask]
#
#         result = {'body_boxes' : torch.cat([match_body_boxes, unmatch_body_boxes]),
#                   'body_scores' : torch.cat([match_body_scores, unmatch_body_scores]),
#
#                   'head_boxes' : match_head_boxes,
#                   'head_scores' :match_head_scores }
#         results.append(result)
#     return results

def body_match(outputs, outputs_head_decoder, thr = 0.01, match_iou_thr = 0.5) :
    results = []
    for output, output_head_decoder in zip(outputs, outputs_head_decoder) :
        scores = output['scores']
        labels = output['labels']
        boxes = output['boxes']

        scores_head_decoder = output_head_decoder['scores']
        labels_head_decoder = output_head_decoder['labels']
        boxes_head_decoder = output_head_decoder['boxes']

        body_mask = (labels == 0) & (scores > thr)
        head_mask = (labels == 1) & (scores > thr)
        # head_mask = (labels == 1) & (scores > thr) & (scores_head_decoder > thr)

        body_boxes = boxes[body_mask]
        body_scores = scores[body_mask]

        head_boxes = boxes[head_mask]
        head_scores = scores[head_mask]
        body_boxes_to_head = boxes_head_decoder[head_mask]

        ious = box_iou(body_boxes, body_boxes_to_head)
        gt_m_idx, dt_m_idx, overlaps = faster_find_argmax(ious, iou_threshlod = match_iou_thr)

        match_body_mask = torch.zeros(len(body_boxes)).bool()
        match_body_mask[gt_m_idx] = True

        unmatch_body_boxes = body_boxes[~match_body_mask]
        unmatch_body_scores = body_scores[~match_body_mask]

        result = {'body_boxes' : torch.cat([body_boxes_to_head, unmatch_body_boxes]),
                  'body_scores' : torch.cat([head_scores, unmatch_body_scores]),

                  'head_boxes' : head_boxes,
                  'head_scores' :head_scores }
        results.append(result)
    return results


def add_gt_two_class_wo_bh(target, gt_body_dict, gt_head_dict, body_gt_ann_id, head_gt_ann_id):
    img_h, img_w = target['orig_size']
    image_info = {'file_name' : target['file_name'],
                  'height' : img_h,
                  'width' : img_w,
                  'id' : target['image_id']}
    gt_body_dict['images'].append(image_info)
    gt_head_dict['images'].append(image_info)

    boxes = target['boxes']
    labels = target['labels']

    for box, label in zip(boxes, labels) :
        if label == 0 :
            xc = box[0] * img_w
            yc = box[1] * img_h
            w = box[2] * img_w
            h = box[3] * img_h

            x1 = xc - w / 2
            y1 = yc - h / 2

            ann = {'area' : w * h,
                   'bbox' : [x1, y1, w, h],
                   'category_id' : 0,
                   'id' : body_gt_ann_id,
                   'ignore' : 0,
                   'image_id' : target['image_id'],
                   'iscrowd' : 0,
                   'segmetation' : []}

            gt_body_dict['annotations'].append(ann)
            body_gt_ann_id = body_gt_ann_id + 1

        if label == 1 :
            xc = box[0] * img_w
            yc = box[1] * img_h
            w = box[2] * img_w
            h = box[3] * img_h

            x1 = xc - w / 2
            y1 = yc - h / 2

            ann = {'area' : w * h,
                   'bbox' : [x1, y1, w, h],
                   'category_id' : 1,
                   'id' : head_gt_ann_id,
                   'ignore' : 0,
                   'image_id' : target['image_id'],
                   'iscrowd' : 0,
                   'segmetation' : []}

            gt_head_dict['annotations'].append(ann)
            head_gt_ann_id = head_gt_ann_id + 1
    return gt_body_dict, gt_head_dict, body_gt_ann_id, head_gt_ann_id

def add_gt_w_bh(target, gt_body_dict, gt_head_dict, gt_match_dict, body_gt_ann_id, head_gt_ann_id, match_gt_ann_id):
    img_h, img_w = target['orig_size']
    image_info = {'file_name' : target['file_name'],
                  'height' : img_h,
                  'width' : img_w,
                  'id' : target['image_id']}
    gt_body_dict['images'].append(image_info)
    gt_head_dict['images'].append(image_info)
    gt_match_dict['images'].append(image_info)

    boxes = target['boxes']
    labels = target['labels']
    annos = target['annotations']
    for box, label in zip(boxes, labels) :
        if label == 1 :
            xc = box[0] * img_w
            yc = box[1] * img_h
            w = box[2] * img_w
            h = box[3] * img_h

            x1 = xc - w / 2
            y1 = yc - h / 2

            ann = {'area' : w * h,
                   'bbox' : [x1, y1, w, h],
                   'category_id' : 0,
                   'id' : body_gt_ann_id,
                   'ignore' : 0,
                   'image_id' : target['image_id'],
                   'iscrowd' : 0,
                   'segmetation' : []}

            gt_body_dict['annotations'].append(ann)
            body_gt_ann_id = body_gt_ann_id + 1

        if label == 2 :
            xc = box[0] * img_w
            yc = box[1] * img_h
            w = box[2] * img_w
            h = box[3] * img_h

            x1 = xc - w / 2
            y1 = yc - h / 2

            ann = {'area' : w * h,
                   'bbox' : [x1, y1, w, h],
                   'category_id' : 1,
                   'id' : head_gt_ann_id,
                   'ignore' : 0,
                   'image_id' : target['image_id'],
                   'iscrowd' : 0,
                   'segmetation' : []}

            gt_head_dict['annotations'].append(ann)
            head_gt_ann_id = head_gt_ann_id + 1

    for anno in annos:
        ann = {'segmentation' : [],
               'vbox': anno['vbox'],
               'area': anno['area'],
               'iscrowd': anno['iscrowd'],
               'height': anno['height'],
               'ignore': anno['ignore'],
               'image_id': target['image_id'],
               'vis_ratio': anno['vis_ratio'],
               'bbox': anno['bbox'],
               'category_id': 1,
               'id' : match_gt_ann_id,
               'h_bbox': anno['h_bbox'],
               'f_bbox': anno['h_bbox']
        }
        gt_match_dict['annotations'].append(ann)
        match_gt_ann_id = match_gt_ann_id + 1

    return gt_body_dict, gt_head_dict, gt_match_dict, body_gt_ann_id, head_gt_ann_id, match_gt_ann_id

def add_gt_one_class(target, gt_dict, gt_ann_id):
    img_h, img_w = target['orig_size']
    image_info = {'file_name' : target['file_name'],
                  'height' : img_h,
                  'width' : img_w,
                  'id' : target['image_id']}
    gt_dict['images'].append(image_info)
    boxes = target['boxes']
    labels = target['labels']

    for box, label in zip(boxes, labels) :
        xc = box[0] * img_w
        yc = box[1] * img_h
        w = box[2] * img_w
        h = box[3] * img_h

        x1 = xc - w / 2
        y1 = yc - h / 2

        ann = {'area' : w * h,
               'bbox' : [x1, y1, w, h],
               'category_id' : 0,
               'id' : gt_ann_id,
               'ignore' : 0,
               'image_id' : target['image_id'],
               'iscrowd' : 0,
               'segmetation' : []}

        gt_dict['annotations'].append(ann)
        gt_ann_id = gt_ann_id + 1
    return gt_dict, gt_ann_id

def filter_out_w_bh(output, thr = 0.01):

    out = {}
    scores = output['scores']
    labels = output['labels']
    boxes = output['boxes']

    body_mask = (labels == 0) & (scores > thr)
    head_mask = (labels == 1) & (scores > thr)

    out['body_boxes'] = boxes[body_mask]
    out['body_scores'] = scores[body_mask]

    out['head_boxes'] = boxes[head_mask]
    out['head_scores'] = scores[head_mask]

    return out

def filter_out_two_class_wo_bh(output, thr = 0.01):

    out = {}
    scores = output['scores']
    labels = output['labels']
    boxes = output['boxes']

    body_mask = (labels == 0) & (scores > thr)
    head_mask = (labels == 1) & (scores > thr)

    out['body_boxes'] = boxes[body_mask]
    out['body_scores'] = scores[body_mask]

    out['head_boxes'] = boxes[head_mask]
    out['head_scores'] = scores[head_mask]

    return out

def filter_out_one_class(output, thr = 0.01):
    out = {}
    scores = output['scores']
    labels = output['labels']
    boxes = output['boxes']
    mask = (labels == 0) & (scores >= thr)
    out['boxes'] = boxes[mask]
    out['scores'] = scores[mask]
    return out

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor] = None):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    @property
    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

    def unmasked_tensor(self, index: int):
        tensor = self.tensors[index]

        if not self.mask[index].any():
            return tensor

        h_index = self.mask[index, 0, :].nonzero(as_tuple=True)[0]
        if len(h_index):
            tensor = tensor[:, :, :h_index[0]]

        w_index = self.mask[index, :, 0].nonzero(as_tuple=True)[0]
        if len(w_index):
            tensor = tensor[:, :w_index[0], :]

        return tensor


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)

        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

    if not is_master:
        def line(*args, **kwargs):
            pass
        def images(*args, **kwargs):
            pass
        Visdom.line = line
        Visdom.images = images


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ and 'SLURM_PTY_PORT' not in os.environ:
        # slurm process but not interactive
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)
    # torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if float(torchvision.__version__[:3]) < 0.7:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


class DistributedWeightedSampler(torch.utils.data.DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, replacement=True):
        super(DistributedWeightedSampler, self).__init__(dataset, num_replicas, rank, shuffle)

        assert replacement

        self.replacement = replacement

    def __iter__(self):
        iter_indices = super(DistributedWeightedSampler, self).__iter__()
        if hasattr(self.dataset, 'sample_weight'):
            indices = list(iter_indices)

            weights = torch.tensor([self.dataset.sample_weight(idx) for idx in indices])

            g = torch.Generator()
            g.manual_seed(self.epoch)

            weight_indices = torch.multinomial(
                weights, self.num_samples, self.replacement, generator=g)
            indices = torch.tensor(indices)[weight_indices]

            iter_indices = iter(indices.tolist())
        return iter_indices

    def __len__(self):
        return self.num_samples


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2, query_mask=None, reduction=True):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs
    # ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")

    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = alpha * ce_loss * ((1 - p_t) ** gamma)

    if not reduction:
        return loss
    return loss.mean(1).sum() / num_boxes

def _sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2, query_mask=None, reduction=True):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy(prob, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = alpha * ce_loss * ((1 - p_t) ** gamma)

    if not reduction:
        return loss
    return loss.mean(1).sum() / num_boxes

def deformable_sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

def nested_dict_to_namespace(dictionary):
    namespace = dictionary
    if isinstance(dictionary, dict):
        namespace = Namespace(**dictionary)
        for key, value in dictionary.items():
            setattr(namespace, key, nested_dict_to_namespace(value))
    return namespace

def nested_dict_to_device(dictionary, device):
    output = {}
    if isinstance(dictionary, dict):
        for key, value in dictionary.items():
            output[key] = nested_dict_to_device(value, device)
        return output
    elif isinstance(dictionary, (str, int, list)):
        return dictionary
    return dictionary.to(device)

def add_dt_wo_bh(dt_body_anns, dt_head_anns, out, img_id):

    body_boxes = out['body_boxes']
    body_scores = out['body_scores']

    head_boxes = out['head_boxes']
    head_scores = out['head_scores']

    for body_box, body_score in zip(body_boxes, body_scores):
        x1, y1, x2, y2 = body_box
        w = x2 - x1
        h = y2 - y1
        ann = {'image_id' : img_id,
               'bbox' : [x1, y1, w, h],
               'score' : body_score,
               'category_id' : 0}
        dt_body_anns.append(ann)

    for head_box, head_score in zip(head_boxes, head_scores):
        x1, y1, x2, y2 = head_box
        w = x2 - x1
        h = y2 - y1
        ann = {'image_id' : img_id,
               'bbox' : [x1, y1, w, h],
               'score' : head_score,
               'category_id' : 1}
        dt_head_anns.append(ann)
    return dt_body_anns, dt_head_anns



def add_dt_w_bh(dt_body_anns, dt_head_anns, out, img_id):

    body_boxes = out['body_boxes']
    body_scores = out['body_scores']

    head_boxes = out['head_boxes']
    head_scores = out['head_scores']

    for body_box, body_score in zip(body_boxes, body_scores):
        x1, y1, x2, y2 = body_box
        w = x2 - x1
        h = y2 - y1
        ann = {'image_id' : img_id,
               'bbox' : [x1, y1, w, h],
               'score' : body_score,
               'category_id' : 0}
        dt_body_anns.append(ann)

    for head_box, head_score in zip(head_boxes, head_scores):
        x1, y1, x2, y2 = head_box
        w = x2 - x1
        h = y2 - y1
        ann = {'image_id' : img_id,
               'bbox' : [x1, y1, w, h],
               'score' : head_score,
               'category_id' : 1}
        dt_head_anns.append(ann)
    return dt_body_anns, dt_head_anns

def add_dt_one_class(dt_anns, out, img_id):
    boxes = out['boxes']
    scores = out['scores']
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        ann = {'image_id' : img_id,
               'bbox' : [x1, y1, w, h],
               'score' : score,
               'category_id' : 0}
        dt_anns.append(ann)
    return dt_anns

def add_match(match_result, out, img_id):
    body_boxes = out['body_boxes']
    body_scores = out['body_scores']

    head_boxes = out['head_boxes']
    head_scores = out['head_scores']

    diff = len(body_boxes) - len(head_boxes)
    if diff > 0 :
        head_boxes = torch.cat([head_boxes, torch.zeros([diff, 4], dtype = torch.float, device = body_boxes.device)])
        head_scores = torch.cat([head_scores, torch.zeros(diff, dtype = torch.float, device = body_boxes.device)])

    for body_box, body_score, head_box, head_score in zip(body_boxes, body_scores, head_boxes, head_scores):
        x1_body, y1_body, x2_body, y2_body = body_box
        w_body = x2_body - x1_body
        h_body = y2_body - y1_body
        x1_head, y1_head, x2_head, y2_head = head_box
        w_head = x2_head - x1_head
        h_head = y2_head - y1_head
        content = {'image_id' : img_id, 'category_id' : 1,
                   'bbox': [x1_body, y1_body, w_body, h_body],
                   'score': body_score,
                   'f_bbox': [x1_head, y1_head, w_head, h_head],
                   'f_score': head_score}
        match_result.append(content)

    return match_result


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            print("MyEncoder-datetime.datetime")
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        if isinstance(obj, int):
            return int(obj)
        elif isinstance(obj, float):
            return float(obj)
        elif isinstance(obj, Tensor):
           return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)