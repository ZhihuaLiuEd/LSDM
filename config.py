import os
import torch

import numpy as np

class Config:
    out_scale=0.001
    exemplar_sz= 128
    instance_sz= 256
    z_size= 127
    x_size= 255
    context= 0.5
    scale_num= 3
    scale_step= 1.0375
    scale_lr= 0.59
    scale_penalty= 0.9745
    window_influence= 0.176
    response_sz= 33
    response_up= 7.75
    total_stride= 4
    epoch_num= 100
    batch_size= 4
    num_workers= 4
    initial_lr= 1e-2
    ultimate_lr=1e-5
    weight_decay= 5e-4
    momentum= 0.99
    r_pos= 32
    r_neg= 0
    c = 16
    k = 64
    iter_num = 3
    def update(self, cfg):
        for k, v in cfg.items():
            setattr(self, k, v)

config=Config()