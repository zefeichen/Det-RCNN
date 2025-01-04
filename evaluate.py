# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import datetime
import os
import random
import sacred
import torch
import yaml
import glob
from argparse import Namespace
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from lib.util.misc import nested_dict_to_namespace, get_rank, collate_fn
from data import build_dataset
from lib.engine import evaluate
from models import build_model
from lib.util.checkpoint_ops import save_checkpoint
import warnings
warnings.filterwarnings('ignore')

ex = sacred.Experiment('val')
ex.add_config(os.path.join(os.getcwd(), 'cfgs/cfg.yaml'))

def main(args: Namespace) -> None :
    torch.backends.cudnn.enable = True
    args.checkpoint_dir = os.path.join(args.root, 'weights/CrowdHuman.pth')

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, criterion, postprocessors = build_model(args)
    model.to(args.device)

    dataset_val = build_dataset(split = 'val', args = args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = DataLoader(
        dataset_val, args.val_cfg.batch_size,
        sampler = sampler_val,
        drop_last = False,
        collate_fn = collate_fn,
        num_workers = args.val_cfg.num_workers)

    checkpoint_dict = torch.load(args.checkpoint_dir, map_location = args.device)
    model.load_state_dict(checkpoint_dict['state_dict'], strict = False)
    evaluate(model, postprocessors, data_loader_val, args)

@ex.main
def load_config(_config, _run) :
    """ We use sacred only for config loading from YAML files. """
    sacred.commands.print_config(_run)

if __name__ == '__main__' :
    config = ex.run_commandline().config
    args = nested_dict_to_namespace(config)
    args.root = os.getcwd()
    main(args)
