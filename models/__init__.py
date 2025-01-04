# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch

from .deformable_DETR import build_deformable_detr

def build_model(args):
    model, criterion, postprocessors = build_deformable_detr(args)
    return model, criterion, postprocessors
