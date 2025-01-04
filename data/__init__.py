# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Submodule interface.
"""
from argparse import Namespace
from pycocotools.coco import COCO
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CocoDetection

from .crowdhuman import build_crowdhuman
from .cityperson import build_citypersons


def get_coco_api_from_dataset(dataset: Subset) -> COCO:
    """Return COCO class from PyTorch dataset for evaluation with COCO eval."""
    for _ in range(10):
        # if isinstance(dataset, CocoDetection):
        #     break
        if isinstance(dataset, Subset):
            dataset = dataset.dataset

    if not isinstance(dataset, CocoDetection):
        raise NotImplementedError

    return dataset.coco


def build_dataset(split: str, args: Namespace) -> Dataset:
    """Helper function to build dataset for different splits ('train' or 'val')."""
    if args.dataset_cfg.dataset == 'CrowdHuman':
        dataset = build_crowdhuman(split, args)
    elif args.dataset_cfg.dataset == 'CityPersons':
        dataset = build_citypersons(split, args)
    else:
        raise ValueError(f'dataset {args.dataset} not supported')
    return dataset
