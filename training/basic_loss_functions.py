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
import sys

# from vif_utils import vif
# from sewar.full_ref import vifp
from skimage.metrics import structural_similarity as ssim

sys.path.append(f'../models')
sys.path.append(f'..')
from models import DAVE2pytorch
from models.DAVE2pytorch import *

################ OG LOSS FUNCTIONS

def loss_fn_orig(recons, x, mu, log_var, kld_weight):
    recons_loss = F.mse_loss(recons, x)
    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
    )
    loss = recons_loss + kld_weight * kld_loss
    return loss, recons_loss, kld_loss

base_model = torch.load("../weights/model-DAVE2v3-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt")
def loss_fn_pred(recons, x, mu, log_var, kld_weight):
    recons_loss = F.mse_loss(recons, x)
    pred_recons = base_model(recons)
    pred_orig = base_model(x)
    prediction_loss = F.mse_loss(pred_orig, pred_recons)
    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
    )
    loss = prediction_loss + recons_loss + kld_weight * kld_loss
    # print(f"{prediction_loss=:.4f}\t{recons_loss=:.4f}\t{(kld_weight * kld_loss)=:.4f}")
    return loss, recons_loss, prediction_loss

################ THIS ONE DOESN'T WORK

def loss_fn_features_kld(recons, x, mu, log_var, kld_weight):
    # recons_loss = vif(recons, x)
    # two identical images have SSIM score of 1
    recons_ssim = recons.clone().detach().cpu().numpy() #.transpose()
    x_ssim = x.clone().detach().cpu().numpy()
    recons_loss = 0
    for i in range(recons_ssim.shape[0]):
        # recons_loss += (1 - ssim(recons_ssim[i], x_ssim[i],
        #               data_range=x_ssim[i].max() - x_ssim[i].min(), win_size=3)) / recons_ssim.shape[0]
        recons_loss += (1 - ssim(recons_ssim[i].transpose(1,2,0), x_ssim[i].transpose(1,2,0))) / recons_ssim.shape[0]
    pred_recons = base_model(recons)
    pred_orig = base_model(x)
    prediction_loss = F.mse_loss(pred_orig, pred_recons)
    # print(f"{prediction_loss=:.4f}   \t{recons_loss=:.4f}")
    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
    )
    loss = prediction_loss + recons_loss + kld_weight * kld_loss
    return loss, recons_loss, prediction_loss


def loss_fn_latent_kld(recons, x, mu, log_var, kld_weight):
    recons_loss = F.mse_loss(recons, x)
    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
    )
    # pred_recons = base_model(recons)
    # pred_orig = base_model(x)
    # pred_loss = F.mse_loss(pred_recons, pred_orig)
    _, latent_recons = base_model.features(recons)
    _, latent_orig = base_model.features(x)
    latent_loss = F.mse_loss(latent_recons, latent_orig)
    loss = recons_loss + latent_loss + kld_weight * kld_loss
    return loss, recons_loss, latent_loss #, pred_loss


def loss_fn_conv(recons, x, mu, log_var, kld_weight, mode='same'):
    # recons_loss = F.mse_loss(recons, x)
    latent_recons, _ = base_model.features(recons)
    latent_orig, _ = base_model.features(x)
    filter_orig = filter(latent_recons, latent_recons)
    filter_pair = filter(latent_recons, latent_orig)
    filter_loss = F.mse_loss(filter_orig, filter_pair)
    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
    )
    pred_recons = base_model(recons)
    pred_orig = base_model(x)
    prediction_loss = F.mse_loss(pred_orig, pred_recons)
    loss = filter_loss + prediction_loss + kld_weight * kld_loss
    return loss, filter_loss, prediction_loss


def filter(img, fltr, mode='same', rot=False):
    if len(img.shape) <= 2:
        img = img[None][None]
    if len(fltr.shape) <= 2:
        fltr = fltr[None][None]
    if rot:
        fltr = torch.rot90(fltr, 2)
    return F.conv2d(img, fltr, padding=mode)