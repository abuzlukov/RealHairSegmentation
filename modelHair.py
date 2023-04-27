# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 09:21:21 2022

@author: anass
"""

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import gc
import ImportData as data_path
from torch.optim import Adam
from matplotlib import pyplot as plt
import utils as utils
import torchvision
from torchvision import transforms

# GPU_CUDA
gc.collect()
T.cuda.empty_cache()


class HairSegmentation(nn.Module):
    def __init__(self,
                enc_chs=(1,64,128,256,512,1024),
                dec_chs=(1024, 512, 256, 128, 64),
                num_class=1,
                retain_dim=False,
                out_sz=(572,572)):
        super().__init__()
        self.encoder     = utils.Encoder(enc_chs)
        self.decoder     = utils.Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out     = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out_sz    = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, out_sz)
        return out