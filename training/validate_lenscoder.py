import shutil
import random
import string
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import time
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
import sys
import threading

# meriel's dependencies
from DatasetGenerator import TransformationDataSequence
# from VIF_loss_functions import *
# from SSIM_loss_functions import *
sys.path.append(f'../models')
from basic_loss_functions import *
from models.DAVE2pytorch import *

import pytorch_fid

class DatasetType(Enum):
    DISJOINT = 1
    INDIST = 2


class LossFn(Enum):
    ORIG = 1
    FEATURES_KLD = 2
    PRED_KLD = 3


def parse_arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument("-t", '--training_dataset', help='parent directory of training dataset')
    # args = parser.parse_args()
    # return args
    parser.add_argument('-v', '--validation_dataset_indist', help='parent directory of in-distribution validation dataset')
    parser.add_argument('-d', '--validation_dataset_disjoint', help='parent directory of disjoint validation dataset')
    parser.add_argument('-o', '--procid', type=int, help='identifier or slurm process id')
    parser.add_argument('-w', '--vae_weights', type=Path, help='Path to trained VAE weights')
    # parser.add_argument('-l', '--loss_fn', type=str, default="orig", help='loss fn type')
    # parser.add_argument('-r', '--lr', type=float, default=0.0001, help='learning rate')
    args = parser.parse_args()
    print("ARGS:", args.training_dataset, args.validation_dataset, args.procid, args.epochs, args.loss_fn, args.lr, flush=True)
    return args


args = parse_arguments()
randstr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
localtime = time.localtime()
timestr = "{}_{}-{}_{}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)
NAME = str(args.vae_weights.parent)
shutil.copy(__file__, NAME)


def validation(vae, dataset, device="cpu", batch=100, robustification=False, noise_level=None):
    validation_dataset = TransformationDataSequence(dataset, image_size=(vae.input_shape[::-1]),
                                            transform=Compose([ToTensor()]), \
                                            robustification=robustification, noise_level=noise_level)
    plot_lock = threading.Lock()
    vae = vae.to(device).eval()
    base_model = torch.load(
        "../weights/model-DAVE2v3-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt").to(device)
    trainloader = DataLoader(validation_dataset, batch_size=batch, shuffle=False)
    transform = Compose([ToTensor()])
    Path(f"{NAME}/validation-sets-{dataset}").mkdir(exist_ok=True, parents=True)
    Path(f"{NAME}/validation-indv-{dataset}").mkdir(exist_ok=True, parents=True)
    for i, hashmap in enumerate(trainloader, start=1):
        with torch.no_grad():
            with plot_lock:
                x = hashmap['image_transf'].float().to(device)
                y = hashmap['image_base'].float().to(device)
                recons, x_out, mu, log_var = vae(x)
                pred_hw1 = base_model(y).detach().cpu().numpy()
                pred_hw2 = base_model(x).detach().cpu().numpy()
                pred_recons = base_model(recons).detach().cpu().numpy()
                # tb_summary(x, recons, pred_orig, pred_recons)
                grid_in = make_grid(x, 10)
                grid_out = make_grid(recons, 10)
                save_image(grid_in, f"{NAME}/validation-sets-{dataset}/input{i:04d}.png")
                save_image(grid_out, f"{NAME}/validation-sets-{dataset}/output{i:04d}.png")
                for j in range(x.shape[0]): #(0, x.shape[0], 10):
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, layout='constrained', sharey=True)
                    hw1_img = np.transpose(y[j].detach().cpu().numpy(), (1, 2, 0))
                    hw2_img = np.transpose(x[j].detach().cpu().numpy(), (1,2,0))
                    recons_img = np.transpose(recons[j].detach().cpu().numpy(), (1,2,0))
                    ax1.imshow(hw1_img)
                    ax2.imshow(hw2_img)
                    ax3.imshow(recons_img)
                    ax1.set_title(f'hw1 pred: {pred_hw1[j][0]:.5f}')
                    ax2.set_title(f'hw2 pred: {pred_hw2[j][0]:.5f}')
                    ax3.set_title(f'recons pred: {pred_recons[j][0]:.5f}')
                    img_name = hashmap['img_name'][j].replace('/', '/\n').replace('\\', '/\n')
                    fig.suptitle(f"{img_name}\n\nHW1-HW2 Prediction error: {(pred_hw1[j][0] - pred_hw2[j][0]):.5f}\n\nHW1-recons Prediction error: {(pred_hw1[j][0] - pred_recons[j][0]):.5f}", fontsize=12)
                    plt.savefig(f"{NAME}/validation-indv-{dataset}/output-batch{i:04d}-sample{j:04d}.png")
                    plt.close(fig)
            if i > 10:
                return


# from torch.utils.tensorboard import SummaryWriter
# def tb_summary(x, recons, pred_orig, pred_recons):
#
#     pass


def main():
    from models.VAEsteer import Model
    print(args, flush=True)
    robustification = False
    noise_level = None
    latent_dim = 512
    model = Model(input_shape=(135,240), latent_dim=latent_dim)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device, flush=True)
    model = torch.load(args.vae_weights).to(device)
    
    # in-distribution validation
    validation(model, args.validation_dataset_indist, device, batch=100)
    
    # disjoint distribution validation
    validation(model, args.validation_dataset_disjoint, device, batch=100)


if __name__ == "__main__":
    main()