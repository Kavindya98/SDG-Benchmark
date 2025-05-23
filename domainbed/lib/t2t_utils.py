# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.

''' 
- load_for_transfer_learning: load pretrained paramters to model in transfer learning
- get_mean_and_std: calculate the mean and std value of dataset.
- msr_init: net parameter initialization.
- progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import torch

import torch.nn as nn
import torch.nn.init as init
import logging
import os
from collections import OrderedDict

_logger = logging.getLogger(__name__)


def resize_pos_embed(posemb, posemb_new): # example: 224:(14x14+1)-> 384: (24x24+1)
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]  # posemb_tok is for cls token, posemb_grid for the following tokens
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))     # 14
    gs_new = int(math.sqrt(ntok_new))             # 24
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)  # [1, 196, dim]->[1, 14, 14, dim]->[1, dim, 14, 14]
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bicubic') # [1, dim, 14, 14] -> [1, dim, 24, 24]
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)   # [1, dim, 24, 24] -> [1, 24*24, dim]
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)   # [1, 24*24+1, dim]
    return posemb

def load_state_dict(checkpoint_path, model, use_ema=False, num_classes=1000, del_posemb=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        if num_classes != 1000:
            # completely discard fully connected for all other differences between pretrained and created model
            del state_dict['head' + '.weight']
            del state_dict['head' + '.bias']

        if del_posemb==True:
            del state_dict['pos_embed']

        old_posemb = state_dict['pos_embed']
        if model.pos_embed.shape != old_posemb.shape:  # need resize the position embedding by interpolate
            new_posemb = resize_pos_embed(old_posemb, model.pos_embed)
            state_dict['pos_embed'] = new_posemb

        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()



def load_for_transfer_learning(model, checkpoint_path, use_ema=False, strict=True, num_classes=1000):
    state_dict = load_state_dict(checkpoint_path, model, use_ema, num_classes)
    model.load_state_dict(state_dict, strict=strict)


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std
