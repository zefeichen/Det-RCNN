# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
CrowdHuman dataset with tracking training augmentations.
"""
from pathlib import Path
from .coco import CocoDetection, make_coco_transforms


def build_citypersons(image_set, args):
    root = Path(args.dataset_cfg.citypersons_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    img_folder = root / f'leftImg8bit/{image_set}'
    ann_file = root / f'body_and_head_annotations/ground-truth/instances_{image_set}_bhfmatch_new.json'
    transforms, norm_transforms = make_coco_transforms(image_set, args.dataset_cfg.img_transform, args.dataset_cfg.overflow_boxes)
    dataset = CocoDetection(img_folder, ann_file, transforms, norm_transforms)
    return dataset
