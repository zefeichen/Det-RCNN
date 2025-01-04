# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import copy
import math
import os
import shutil
import time
from typing import Iterable
import torch
from lib.evaluate import compute_MMR
from lib.util.vis import vis_dataload, plot_boxes_wo_bh, plot_boxes_w_bh_head_decoder, plot_boxes_w_bh

from lib.util import misc as utils
from lib.util.misc import MetricLogger, nested_dict_to_device, SmoothedValue
import json
import sys
from thop import profile
from lib.util.misc import body_match, add_match, add_dt_wo_bh, add_gt_w_bh,  filter_out_w_bh,  add_dt_w_bh
from lib.util.misc import MyEncoder
from lib.evaluate.COCOEVAL import run_evaluate

def train_one_epoch(model: torch.nn.Module, criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, args) :
    model.train()
    record_file = os.path.join(args.checkpoint_dir, args.train_cfg.record_file)
    metric_logger = MetricLogger(args.train_cfg.vis_and_log_interval, delimiter = " ; ", record_file = record_file)
    metric_logger.add_meter('lr', SmoothedValue(window_size = 1, fmt = '{value:.2e}'))
    metric_logger.add_meter('batch_size', SmoothedValue(window_size = 1, fmt = '{value:2d}'))

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, epoch)) :
        if args.train_cfg.vis_train_data :
            font_path = os.path.join(args.root, args.font_path)
            vis_train_data_path = os.path.join(args.root, args.train_cfg.vis_train_data_path)
            vis_dataload(samples, targets, path = vis_train_data_path, font_path = font_path)

        samples = samples.to(device)
        targets = [nested_dict_to_device(t, device) for t in targets]

        outputs, outputs_head_decoder = model(samples)

        loss_dict = criterion(outputs, outputs_head_decoder, targets)

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f'{k}_unscaled' : v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {
            k : v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if args.train_cfg.clip_max_norm > 0 :
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.train_cfg.clip_max_norm)
        optimizer.step()

        metric_logger.update(loss = loss_value,
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)

        metric_logger.update(lr = optimizer.param_groups[0]["lr"])
        metric_logger.update(batch_size = args.train_cfg.batch_size)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k : meter.global_avg for k, meter in metric_logger.meters.items()}

class GT:
    def __init__(self):
        self.gt_body_dict = {'type' : 'instances',
                             'categories' : [{'id' : 0, 'name' : 'body', 'supercategory' : 'none'}],
                             'images' : [],
                             'annotations' : []}

        self.gt_head_dict = {'type' : 'instances',
                             'categories' : [{'id' : 0, 'name' : 'head', 'supercategory' : 'none'}],
                             'images' : [],
                             'annotations' : []}

        self.gt_head_dict_ = {'type' : 'instances',
                             'categories' : [{'id' : 1, 'name' : 'head', 'supercategory' : 'none'}],
                             'images' : [],
                             'annotations' : []}

        self.match_dict = {'images':[],
                           'annotations':[],
                           'categories': [{'id' : 1, 'name' : 'person'}]}

    def get_dict(self, body, head):
        if body and head:
            return self.gt_body_dict, self.gt_head_dict_
        elif body and not head:
            return self.gt_body_dict
        elif not body and head:
            return self.gt_head_dict

def evaluate(model, postprocessors, data_loader_val, args):
    model.eval()
    save_path = f'{args.root}/{args.save_path}/val'
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        os.makedirs(save_path)
    else :
        os.makedirs(save_path)

    evaluate_result_file = os.path.join(save_path, args.val_cfg.evaluate_result_file)

    gt_body_path = os.path.join(save_path, 'gt_body.json')
    gt_head_path = os.path.join(save_path, 'gt_head.json')

    orig_dt_body_path = os.path.join(save_path, 'orig_dt_body.json')
    orig_dt_head_path = os.path.join(save_path, 'orig_dt_head.json')

    match_dt_body_path = os.path.join(save_path, 'match_dt_body.json')
    match_dt_head_path = os.path.join(save_path, 'match_dt_head.json')

    fpath_match = os.path.join(save_path, 'match.json')
    gt_fpath_match = os.path.join(save_path, 'gt_match.json')

    metric_logger = MetricLogger(args.val_cfg.vis_and_log_interval, delimiter = " ; ")

    body_gt_ann_id = 0
    head_gt_ann_id = 0
    match_gt_ann_id = 1

    orig_dt_body_anns = []
    orig_dt_head_anns = []

    match_dt_body_anns = []
    match_dt_head_anns = []

    gt = GT()
    gt_body_dict, gt_head_dict = gt.get_dict(args.model_cfg.body, args.model_cfg.head)
    gt_match_dict = gt.match_dict
    match_result = []
    font_path = os.path.join(args.root, args.font_path)
    image_path = os.path.join(args.root, 'DataSets/CrowdHuman/Images/val')
    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader_val, 'Test')):
        samples = samples.to(args.device)
        targets = [nested_dict_to_device(t, args.device) for t in targets]
        with torch.no_grad() :
            outputs, outputs_head_decoder = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim = 0)
        outputs = postprocessors['bbox'](outputs, orig_target_sizes)
        outputs_head_decoder = postprocessors['bbox'](outputs_head_decoder, orig_target_sizes)

        for output, target in zip(outputs, targets) :
            gt_body_dict, gt_head_dict, gt_match_dict, body_gt_ann_id, head_gt_ann_id, match_gt_ann_id = add_gt_w_bh(target, gt_body_dict,
                                                                                                                     gt_head_dict, gt_match_dict,
                                                                                                                     body_gt_ann_id, head_gt_ann_id, match_gt_ann_id)
            out = filter_out_w_bh(output, thr = args.model_cfg.evaluate_thr)
            orig_dt_body_anns, orig_dt_head_anns = add_dt_wo_bh(orig_dt_body_anns, orig_dt_head_anns, out, target['image_id'])
        if args.val_cfg.plot_boxes :
            plot_boxes_wo_bh(out, target, image_path, save_path, font_path)
            plot_boxes_w_bh_head_decoder(outputs, outputs_head_decoder, targets, image_path, save_path, font_path)
        outs = body_match(outputs, outputs_head_decoder, thr = args.model_cfg.evaluate_thr, match_iou_thr = args.model_cfg.match_iou_thr)
        for out, target in zip(outs, targets) :
            match_dt_body_anns, match_dt_head_anns = add_dt_w_bh(match_dt_body_anns, match_dt_head_anns, out,
                                                                 target['image_id'])
            match_result = add_match(match_result, out, target['image_id'])
            if args.val_cfg.plot_boxes :
                plot_boxes_w_bh(out, target, image_path, save_path, font_path)

    with open(gt_body_path, 'w', encoding = 'utf-8') as f :
        json.dump(gt_body_dict, f, cls = MyEncoder, ensure_ascii = False)

    with open(gt_head_path, 'w', encoding = 'utf-8') as f :
        json.dump(gt_head_dict, f, cls = MyEncoder, ensure_ascii = False)

    with open(gt_fpath_match, 'w', encoding = 'utf-8') as f :
        json.dump(gt_match_dict, f, cls = MyEncoder, ensure_ascii = False)

    with open(orig_dt_body_path, 'w', encoding = 'utf-8') as f :
        json.dump(orig_dt_body_anns, f, cls = MyEncoder, ensure_ascii = False)

    with open(orig_dt_head_path, 'w', encoding = 'utf-8') as f :
        json.dump(orig_dt_head_anns, f, cls = MyEncoder, ensure_ascii = False)

    with open(match_dt_body_path, 'w', encoding = 'utf-8') as f :
        json.dump(match_dt_body_anns, f, cls = MyEncoder, ensure_ascii = False)

    with open(match_dt_head_path, 'w', encoding = 'utf-8') as f :
        json.dump(match_dt_head_anns, f, cls = MyEncoder, ensure_ascii = False)

    with open(fpath_match, 'w', encoding = 'utf-8') as f :
        json.dump(match_result, f, cls = MyEncoder, ensure_ascii = False)

    print('file have been saved')

    start_info = ''
    start_info += f'{"*" * 40}ORIG BODY{"*" * 40}'
    with open(evaluate_result_file, 'a+') as file :
        print(start_info, file = file)
    orig_body_cocoEval = run_evaluate(gt_body_path, orig_dt_body_path, evaluate_result_file)

    with open(evaluate_result_file, 'a+') as file :
        print(f'{"*" * 40}ORIG HEAD{"*" * 40}', file = file)
    orig_head_cocoEval = run_evaluate(gt_head_path, orig_dt_head_path, evaluate_result_file)

    with open(evaluate_result_file, 'a+') as file :
        print(f'{"*" * 100}', file = file)

    line = f'{"*" * 40}MATCH BODY{"*" * 40}'
    with open(evaluate_result_file, 'a+') as file :
        print(line, file = file)
    match_body_cocoEval = run_evaluate(gt_body_path, match_dt_body_path, evaluate_result_file)

    line = f'{"*" * 40}MATCH HEAD{"*" * 40}'
    with open(evaluate_result_file, 'a+') as file :
        print(line, file = file)
    match_head_cocoEval = run_evaluate(gt_head_path, match_dt_head_path, evaluate_result_file)

    MMR = compute_MMR.compute_MMR(fpath_match, gt_fpath_match, record_file = evaluate_result_file)