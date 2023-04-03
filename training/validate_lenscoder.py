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
# import pytorch_fid

# meriel's dependencies
from DatasetGenerator import TransformationDataSequence
# from VIF_loss_functions import *
# from SSIM_loss_functions import *
sys.path.append(f'../')
sys.path.append(f'../models')
from models.DAVE2pytorch import *


class DatasetType(Enum):
    DISJOINT = 1
    INDIST = 2


class LossFn(Enum):
    ORIG = 1
    FEATURES_KLD = 2
    PRED_KLD = 3


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--validation_dataset_indist', help='parent directory of in-distribution validation dataset')
    parser.add_argument('-d', '--validation_dataset_disjoint', help='parent directory of disjoint validation dataset')
    parser.add_argument('-o', '--procid', type=int, help='identifier or slurm process id')
    parser.add_argument('-w', '--vae_weights', type=Path, help='Path to trained VAE weights')
    args = parser.parse_args()
    print("ARGS:", args.validation_dataset_indist, args.validation_dataset_disjoint, args.procid, args.vae_weights, flush=True)
    return args


args = parse_arguments()
randstr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
localtime = time.localtime()
timestr = "{}_{}-{}_{}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)
NAME = str(args.vae_weights.parent)+"_TEST2"
Path(NAME).mkdir(exist_ok=True, parents=True)
shutil.copy(__file__, NAME)

# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/figure_title.html
def validation_viz(vae, dataset, device="cpu", batch=100, shuffle=False, sample=False):
    import threading
    plot_lock = threading.Lock()
    vae = vae.to(device).eval()
    base_model = torch.load(
        "../weights/model-DAVE2v3-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt",
        map_location=device
    )
    trainloader = DataLoader(dataset, batch_size=batch, shuffle=shuffle)
    # transform = Compose([ToTensor()])
    Path(f"samples_{NAME}/validation-sets").mkdir(exist_ok=True, parents=True)
    Path(f"samples_{NAME}/validation-indv").mkdir(exist_ok=True, parents=True)
    if sample:
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
                    save_image(grid_in, f"samples_{NAME}/validation-sets/input{i:04d}.png")
                    save_image(grid_out, f"samples_{NAME}/validation-sets/output{i:04d}.png")
                    for j in range(0, x.shape[0], 10):
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
                        plt.savefig(f"samples_{NAME}/validation-indv/output-batch{i:04d}-sample{j:04d}.png")
                        plt.close(fig)
                if i > 5:
                    return
    else:
        for i, hashmap in enumerate(trainloader, start=1):
                    with torch.no_grad():
                        with plot_lock:
                            x = hashmap['image_transf'].float().to(device)
                            y = hashmap['image_base'].float().to(device)
                            sample_name = hashmap['img_name']
                            recons, x_out, mu, log_var = vae(x)
                            pred_hw1 = base_model(y).detach().cpu().numpy()
                            pred_hw2 = base_model(x).detach().cpu().numpy()
                            pred_recons = base_model(recons).detach().cpu().numpy()
                            # tb_summary(x, recons, pred_orig, pred_recons)
                            grid_in = make_grid(x, 10)
                            grid_out = make_grid(recons, 10)
                            save_image(grid_in, f"samples_{NAME}/validation-sets/input{i:04d}.png")
                            save_image(grid_out, f"samples_{NAME}/validation-sets/output{i:04d}.png")
                            for j in range(0, x.shape[0], 10):
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
                                sample_name_crop = str(Path(sample_name[j]).name).replace("jpg", "")
                                fig.suptitle(f"{img_name}\n\nHW1-HW2 Prediction error: {(pred_hw1[j][0] - pred_hw2[j][0]):.5f}\n\nHW1-recons Prediction error: {(pred_hw1[j][0] - pred_recons[j][0]):.5f}", fontsize=12)
                                plt.savefig(f"samples_{NAME}/validation-indv/output-{sample_name_crop}-batch{i:04d}-sample{j:04d}.png")
                                plt.close(fig)


def validation(vae, dataset, device="cpu", batch=100):
    vae = vae.to(device).eval()
    base_model = torch.load(
        "../weights/model-DAVE2v3-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt",
        map_location=device
    )
    trainloader = DataLoader(dataset, batch_size=batch, shuffle=False)
    errors_recons = torch.zeros(0).to(device) #(dataset.size, 1))
    errors_hw2 = torch.zeros(0).to(device)
    temp = torch.zeros(0).to(device)
    for i, hashmap in enumerate(trainloader, start=1):
        with torch.no_grad():
            x = hashmap['image_transf'].float().to(device)
            y = hashmap['image_base'].float().to(device)
            recons, x_out, mu, log_var = vae(x)
            pred_hw1 = base_model(y)
            pred_hw2 = base_model(x)
            pred_recons = base_model(recons)
            # tb_summary(x, recons, pred_orig, pred_recons)
            torch.cat((errors_recons, pred_hw1 - pred_recons), out=temp)
            errors_recons = torch.clone(temp)
            torch.cat((errors_hw2, pred_hw1 - pred_hw2), out=temp)
            errors_hw2 = torch.clone(temp)
    # mae_recons = torch.sum(errors_recons) / dataset.size
    recons = torch.std_mean(errors_recons)
    # mae_hw2 = torch.sum(errors_hw2) / dataset.size
    hw2 = torch.std_mean(errors_hw2)
    print("recons error:", recons)
    print("hw2 error:", hw2)
    return recons, hw2


def main():
    from models.VAEsteer import Model
    print(args, flush=True)
    latent_dim = 512
    model = Model(input_shape=(135,240), latent_dim=latent_dim)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device, flush=True)
    model = torch.load(args.vae_weights).to(device)
    
    # in-distribution validation
    validation_dataset_indist = TransformationDataSequence(args.validation_dataset_indist, image_size=(model.input_shape[::-1]),
                                                transform=Compose([ToTensor()]))
    validation(model, validation_dataset_indist, device, batch=100)
    validation_viz(model, validation_dataset_indist, device, batch=100, shuffle=False, sample=False)
    
    # disjoint distribution validation
    validation_dataset_disj = TransformationDataSequence(args.validation_dataset_disjoint, image_size=(model.input_shape[::-1]),
                                                  transform=Compose([ToTensor()]))
    validation(model, validation_dataset_disj, device, batch=100)
    validation_viz(model, validation_dataset_disj, device, batch=100, shuffle=True, sample=True)


if __name__ == "__main__":
    main()