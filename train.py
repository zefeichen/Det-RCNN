import datetime
import os
import random
import time
import sacred
import torch
import yaml
import glob
import torch.optim as optim
from argparse import Namespace
import numpy as np
from torch.utils.data import DataLoader
from lib.util.checkpoint_ops import save_checkpoint
from lib.util.misc import nested_dict_to_namespace, get_rank, collate_fn
from data import build_dataset
from lib.engine import train_one_epoch, evaluate
from models import build_model
import warnings
warnings.filterwarnings('ignore')

ex = sacred.Experiment('train')
ex.add_config(os.path.join(os.getcwd(), 'cfgs/cfg.yaml'))


def train(args: Namespace) -> None :
    torch.backends.cudnn.enable = True
    args.checkpoint_dir = os.path.join(args.root, args.checkpoint_dir)
    if not os.path.exists(args.checkpoint_dir) :
        os.makedirs(args.checkpoint_dir)
    yaml.dump(vars(args), open(os.path.join(args.checkpoint_dir, 'config.yaml'), 'w'), allow_unicode = True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device
    # fix the seed for reproducibility
    seed = args.train_cfg.seed + get_rank()
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('NUM TRAINABLE MODEL PARAMS:', n_parameters)

    params = [p for n, p in model.named_parameters()]

    optimizer_type = {
        'adam' : torch.optim.Adam(params, lr = args.train_cfg.lr, weight_decay = args.train_cfg.weight_decay),
        'sgd' : torch.optim.SGD(params, args.train_cfg.lr, momentum = args.train_cfg.beta1,
                                weight_decay = args.train_cfg.weight_decay),
        'adagrad' : torch.optim.Adagrad(params, args.train_cfg.lr, weight_decay = args.train_cfg.weight_decay),
        'rmsprop' : torch.optim.RMSprop(params, args.train_cfg.lr, alpha = 0.99, eps = 1e-08,
                                        weight_decay = args.train_cfg.weight_decay, momentum = args.train_cfg.beta1,
                                        centered = False),
        'asgd' : torch.optim.ASGD(params, lr = args.train_cfg.lr, lambd = 0.0001, alpha = 0.75, t0 = 1000000.0,
                                  weight_decay = args.train_cfg.weight_decay),
        'adadelta' : torch.optim.Adadelta(params, lr = args.train_cfg.lr, rho = 0.9, eps = 1e-06,
                                          weight_decay = args.train_cfg.weight_decay),
        'adamax' : torch.optim.Adamax(params, lr = args.train_cfg.lr, betas = (0.9, 0.999), eps = 1e-08,
                                      weight_decay = args.train_cfg.weight_decay),
        'adamW' : torch.optim.AdamW(params, lr = args.train_cfg.lr,
                                    betas = (args.train_cfg.beta1, args.train_cfg.beta2),
                                    weight_decay = args.train_cfg.weight_decay, amsgrad = args.train_cfg.amsgrad)
    }

    optimizer = optimizer_type[args.train_cfg.optimizer_type]
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = args.train_cfg.step_size,
                                             gamma = args.train_cfg.gamma)

    dataset_train = build_dataset(split = 'train', args = args)
    dataset_val = build_dataset(split = 'val', args = args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.train_cfg.batch_size, drop_last = True)

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler = batch_sampler_train,
        collate_fn = collate_fn,
        num_workers = args.train_cfg.num_workers,
        pin_memory = False)

    data_loader_val = DataLoader(
        dataset_val, args.val_cfg.batch_size,
        sampler = sampler_val,
        drop_last = False,
        collate_fn = collate_fn,
        num_workers = args.val_cfg.num_workers,
        pin_memory = False)

    print("Start training")
    start_time = time.time()
    start_epoch = args.train_cfg.start_epoch
    epochs = args.train_cfg.epochs

    checkpoint_path_list = sorted(glob.glob(os.path.join(args.checkpoint_dir, 'checkpoint_ep*.pth')))
    if len(checkpoint_path_list) > 0:
        checkpoint_path = checkpoint_path_list[-1]
        print(checkpoint_path, ' has been loaded')
        checkpoint_dict = torch.load(checkpoint_path, map_location = device)
        model.load_state_dict(checkpoint_dict['state_dict'], strict = False)
        start_epoch = checkpoint_dict['epoch'] + 1
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
        lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
    else :
        print('None of checkpoints is been loaded')

    for epoch in range(start_epoch, epochs + 1) :
        checkpoint_dict = {}
        # train
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, args)
        lr_scheduler.step()
        checkpoint_dict['state_dict'] = model.state_dict()
        save_checkpoint(checkpoint_dict, args.checkpoint_dir)

        # evaluate
        if epoch == 1 or (epoch % args.val_cfg.val_interval == 0):
            evaluate(model, postprocessors, data_loader_val, args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds = int(total_time)))
    print('Training time {}'.format(total_time_str))


@ex.main
def load_config(_config, _run) :
    """ We use sacred only for config loading from YAML files. """
    sacred.commands.print_config(_run)

if __name__ == '__main__' :
    config = ex.run_commandline().config
    args = nested_dict_to_namespace(config)
    args.root = os.getcwd()
    train(args)