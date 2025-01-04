# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch

from .backbone import build_backbone
from .deformable_detr import DeformableDETR, DeformablePostProcess
from .deformable_transformer import build_deforamble_transformer
from .detr import PostProcess, SetCriterion
from .matcher import build_matcher
from .build_trackformer import build_trackformer

def build_deformable_detr(args):
    num_classes = 1 + args.model_cfg.body + args.model_cfg.head
    args.num_classes = num_classes
    matcher = build_matcher(args)
    weight_dict = {'loss_ce' : args.deformable_detr.loss_coef.cls_loss_coef,
                   'loss_bbox' : args.deformable_detr.loss_coef.bbox_loss_coef,
                   'loss_giou' : args.deformable_detr.loss_coef.giou_loss_coef, }

    if args.model_cfg.aux_loss :
        aux_weight_dict = {}
        for i in range(args.deformable_detr.dec_layers - 1) :
            aux_weight_dict.update({k + f'_{i}' : v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    weight_dict_second = {'loss_head_decoder_ce' : args.deformable_detr.loss_coef.cls_loss_head_decoder_coef,
                          'loss_head_decoder_bbox' : args.deformable_detr.loss_coef.bbox_loss_head_decoder_coef,
                          'loss_head_decoder_giou' : args.deformable_detr.loss_coef.giou_loss_head_decoder_coef}
    if args.model_cfg.aux_loss :
        aux_weight_dict = {}
        for i in range(args.deformable_detr.dec_layers - 1) :
            aux_weight_dict.update({k + f'_{i}' : v for k, v in weight_dict_second.items()})
        weight_dict_second.update(aux_weight_dict)

    weight_dict.update(weight_dict_second)

    losses = ['labels', 'boxes', 'cardinality']
    head_decoder_losses = ['head_decoder_labels', 'head_decoder_boxes']
    criterion = SetCriterion(num_classes, matcher, args, weight_dict, losses, head_decoder_losses, args.model_cfg.bh,
                             focal_alpha = args.deformable_detr.focal_loss.focal_alpha,
                             focal_gamma = args.deformable_detr.focal_loss.focal_gamma)

    postprocessors = {'bbox' : PostProcess()}

    backbone = build_backbone(args)
    transformer = build_deforamble_transformer(args)
    deformable_detr_kwargs = {
        'backbone': backbone,
        'num_classes': args.num_classes,
        'num_queries': args.model_cfg.num_queries,
        'bh' : args.model_cfg.bh,
        'aux_loss': args.model_cfg.aux_loss,
        'with_box_refine': args.model_cfg.with_box_refine,
        'overflow_boxes': args.dataset_cfg.overflow_boxes,
        'num_feature_levels': args.deformable_detr.num_feature_levels,
        'head_query_false_positive_prob' : args.deformable_detr.head_query_false_positive_prob,
        'transformer': transformer}
    model = DeformableDETR(**deformable_detr_kwargs)

    return model, criterion, postprocessors
