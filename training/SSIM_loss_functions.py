import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Compose, ToTensor, Resize, Lambda, Normalize
from torch.autograd import Variable
import argparse
from pathlib import Path
from torchvision.utils import make_grid, save_image
from scipy import signal
from enum import Enum

from vif_utils import vif
from skimage.metrics import structural_similarity as ssim
from sewar.full_ref import vifp

class SSIMLoss():
    def __init__(self):
        pass

    def structural_similarity(self, im1, im2,
                              *,
                              win_size=None, gradient=False, data_range=None,
                              channel_axis=0,
                              gaussian_weights=False, full=False, **kwargs):

        # check_shape_equality(im1, im2)
        float_type = torch.float32

        # loop over channels
        args = dict(win_size=win_size,
                    gradient=gradient,
                    data_range=data_range,
                    channel_axis=None,
                    gaussian_weights=gaussian_weights,
                    full=full)
        args.update(kwargs)
        nch = im1.shape[channel_axis]
        mssim = torch.empty(nch, dtype=float_type)

        if gradient:
            G = torch.empty(im1.shape, dtype=float_type)
        if full:
            S = torch.empty(im1.shape, dtype=float_type)
        channel_axis = channel_axis % im1.ndim
        _at = self.partial(self.slice_at_axis, axis=channel_axis)
        for ch in range(nch):
            ch_result = self.structural_similarity(im1[_at(ch)],
                                              im2[_at(ch)], **args)
            if gradient and full:
                mssim[ch], G[_at(ch)], S[_at(ch)] = ch_result
            elif gradient:
                mssim[ch], G[_at(ch)] = ch_result
            elif full:
                mssim[ch], S[_at(ch)] = ch_result
            else:
                mssim[ch] = ch_result
        mssim = mssim.mean()
        if gradient and full:
            return mssim, G, S
        elif gradient:
            return mssim, G
        elif full:
            return mssim, S
        else:
            return mssim

    def partial(self):
        pass

    def slice_at_axis(self):
        pass

