
import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
import copy

from types import SimpleNamespace

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed import model_selection
from domainbed.lib.query import Q

import os

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str, default="./domainbed/data")
    parser.add_argument('--dataset', type=str, default="PACS")
    parser.add_argument('--algorithm', type=str, default="ViT_RB_small")
    parser.add_argument('--task', type=str, default="domain_generalization",
                        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
                        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
                        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
                        help='Trial number (used for seeding split_dataset and '
                             'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
                        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
                        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="test_ViT_RB")
    parser.add_argument('--holdout_fraction', type=float, default=0.001)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
                        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--save_best_model', action='store_true')

    args = parser.parse_args()
    args.save_best_model = True

    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
                                                  misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        js = json.loads(args.hparams)
        js["test_env"] = args.test_envs
        # print(args.hparams)
        hparams.update(js)

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print('device:', device)
    print ('Current cuda device ', torch.cuda.current_device())

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
                                               args.test_envs, hparams)
    else:
        raise NotImplementedError

in_splits = []
out_splits = []
uda_splits = []
assigned = False
for env_i, env in enumerate(dataset):  # env is a domain
    
    if  hparams['custom_train_val'] and (hparams['custom_val'] == env_i or hparams['custom_train'] == env_i):
        if not assigned:
            for env_j, env_ in enumerate(dataset):
            
                if  hparams['custom_val'] == env_j:
                    out = env_
                elif  hparams['custom_train'] == env_j:
                    in_= env_
            assigned = True
            print(len(out)," length of val ",len(in_)," length of train") 
        else:
            continue  
    
        
data_loader_train = torch.utils.data.DataLoader(
        in_, sampler=torch.utils.data.RandomSampler(in_),
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS,
    )

data_loader_val = torch.utils.data.DataLoader(
        out, sampler=torch.utils.data.SequentialSampler(out),
        batch_size=int(1.5 * hparams['batch_size']),
        num_workers=dataset.N_WORKERS,
    )


mixup_fn = Mixup(
            mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
            prob=1.0, switch_prob=0.5, mode='batch',
            label_smoothing=0.1, num_classes=dataset.num_classes)

model = create_model(
        'resnet50',
        pretrained=False,
        num_classes=dataset.num_classes,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size=224
    )

args1 = SimpleNamespace()
args1.weight_decay = 1e-4
args1.lr = 0.1
args1.opt = 'sgd' #'lookahead_adam' to use `lookahead`
args1.momentum = 0.9
args1.sched='step'
args1.decay_t = 30

optimizer = create_optimizer(args, model)
