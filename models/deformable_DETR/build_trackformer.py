# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch

from .backbone import build_backbone
from .deformable_detr import DeformableDETR, DeformablePostProcess
from .deformable_transformer import build_deforamble_transformer
from .detr import DETR, PostProcess, SetCriterion
from .matcher import build_matcher


def build_trackformer(args):

    num_classes = 20

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone = build_backbone(args)
    matcher = build_matcher(args)

    detr_kwargs = {
        'backbone': backbone,
        'num_classes': num_classes - 1 if args.trackformer.focal_loss else num_classes,
        'num_queries': args.trackformer.num_queries,
        'aux_loss': args.trackformer.aux_loss,
        'overflow_boxes': args.trackformer.overflow_boxes}

    tracking_kwargs = {
        'track_query_false_positive_prob': args.trackformer.track_query_false_positive_prob,
        'track_query_false_negative_prob': args.trackformer.track_query_false_negative_prob,
        'matcher': matcher,
        'backprop_prev_frame': args.trackformer.track_backprop_prev_frame,}


    transformer = build_deforamble_transformer(args)

    detr_kwargs['transformer'] = transformer
    detr_kwargs['num_feature_levels'] = args.trackformer.num_feature_levels
    detr_kwargs['with_box_refine'] = args.trackformer.with_box_refine
    detr_kwargs['two_stage'] = args.two_stage
    detr_kwargs['multi_frame_attention'] = args.multi_frame_attention
    detr_kwargs['multi_frame_encoding'] = args.multi_frame_encoding
    detr_kwargs['merge_frame_features'] = args.merge_frame_features

    model = DeformableDETRTracking(tracking_kwargs, detr_kwargs)

    weight_dict = {'loss_ce': args.cls_loss_coef,
                   'loss_bbox': args.bbox_loss_coef,
                   'loss_giou': args.giou_loss_coef,}

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})

        if args.two_stage:
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']

    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        focal_loss=args.focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        tracking=args.tracking,
        track_query_false_positive_eos_weight=args.track_query_false_positive_eos_weight,)
    criterion.to(device)

    if args.focal_loss:
        postprocessors = {'bbox': DeformablePostProcess()}
    else:
        postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors
