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
NAME = f"{args.vae_weights.parent}" #_VALIDATION_{args.procid}"
Path(NAME).mkdir(exist_ok=True, parents=True)
shutil.copy(__file__, NAME)
print(f"Running {__file__}")
print(f"Logging to {NAME}")


def validation_viz(vae, dataset, device="cpu", batch=100, shuffle=False, sample=False, dataset_type=""):
    import threading
    plot_lock = threading.Lock()
    vae = vae.to(device).eval()
    base_model = torch.load(
        "../weights/model-DAVE2v3-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt",
        map_location=device
    )
    trainloader = DataLoader(dataset, batch_size=batch, shuffle=shuffle)
    # transform = Compose([ToTensor()])
    Path(f"{NAME}/validation-sets-{dataset_type}").mkdir(exist_ok=True, parents=True)
    Path(f"{NAME}/validation-indv-{dataset_type}").mkdir(exist_ok=True, parents=True)
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
                    save_image(grid_in, f"{NAME}/validation-sets-{dataset_type}/input{i:04d}.png")
                    save_image(grid_out, f"{NAME}/validation-sets-{dataset_type}/output{i:04d}.png")
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
                        plt.savefig(f"{NAME}/validation-indv-{dataset_type}/output-batch{i:04d}-sample{j:04d}.png")
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
                            save_image(grid_in, f"{NAME}/validation-sets-{dataset_type}/input{i:04d}.png")
                            save_image(grid_out, f"{NAME}/validation-sets-{dataset_type}/output{i:04d}.png")
                            for j in range(x.shape[0]):
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
                                plt.savefig(f"{NAME}/validation-indv-{dataset_type}/output-{sample_name_crop}-batch{i:04d}-sample{j:04d}.png")
                                plt.close(fig)


def viz_biggest_hw2_errors(vae, dataset, device, batch=100, dataset_type=""):
    import threading
    plot_lock = threading.Lock()
    vae = vae.to(device).eval()
    base_model = torch.load(
        "../weights/model-DAVE2v3-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt",
        map_location=device
    )
    trainloader = DataLoader(dataset, batch_size=batch, shuffle=False)
    Path(f"{NAME}/validation-indv-{dataset_type}-largesthw2errors").mkdir(exist_ok=True, parents=True)
    for i, hashmap in enumerate(trainloader, start=1):
        with torch.no_grad():
            x = hashmap['image_transf'].float().to(device)
            y = hashmap['image_base'].float().to(device)
            # sample_names = hashmap['img_name']
            recons, x_out, mu, log_var = vae(x)
            pred_hw1 = base_model(y).detach().cpu().numpy()
            pred_hw2 = base_model(x).detach().cpu().numpy()
            pred_recons = base_model(recons).detach().cpu().numpy()
            for j in range(x.shape[0]):
                sample_name = hashmap['img_name'][j]
                if "01608" in sample_name or "01063" in sample_name or "01464" in sample_name or "01052" in sample_name or "00249" in sample_name or "01630" in sample_name:
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
                    sample_name_crop = str(Path(hashmap['img_name'][j]).name).replace(".jpg", "")
                    fig.suptitle(f"{img_name}\n\nHW1-HW2 Prediction error: {(pred_hw1[j][0] - pred_hw2[j][0]):.5f}\n\nHW1-recons Prediction error: {(pred_hw1[j][0] - pred_recons[j][0]):.5f}", fontsize=12)
                    plt.savefig(f"{NAME}/validation-indv-{dataset_type}-largesthw2errors/output-{sample_name_crop}.png")
                    plt.close(fig)


def get_biggest_hw2_errors(vae, dataset, device, batch=100):
    import collections
    # In [2]: d = {2:3, 1:89, 4:5, 3:0}
    # In [3]: od = collections.OrderedDict(sorted(d.items()))
    # In [4]: od
    # Out[4]: OrderedDict([(1, 89), (2, 3), (3, 0), (4, 5)])
    vae = vae.to(device).eval()
    base_model = torch.load(
        "../weights/model-DAVE2v3-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt",
        map_location=device
    )
    trainloader = DataLoader(dataset, batch_size=batch, shuffle=False)
    errors_hw2 = torch.zeros(0).to(device)
    five_largest = {}
    od = collections.OrderedDict(sorted(five_largest.items()))
    for i, hashmap in enumerate(trainloader, start=1):
        with torch.no_grad():
            x = hashmap['image_transf'].float().to(device)
            y = hashmap['image_base'].float().to(device)
            pred_hw1 = base_model(y)
            pred_hw2 = base_model(x)
            errors_hw2 = torch.abs(pred_hw1 - pred_hw2)
            for j in range(x.shape[0]):
                hw2_error = errors_hw2[j].numpy()[0]
                five_largest[hw2_error] = hashmap['img_name'][j]
    od = collections.OrderedDict(sorted(five_largest.items()))
    with open("hw2_errors.txt", "w") as f:
        for key in od.keys():
            print(key, od[key])
            f.write(f"{key},{od[key]}\n")


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
    recons = torch.std_mean(errors_recons)
    recons_abs = torch.std_mean(torch.abs(errors_recons))
    hw2 = torch.std_mean(errors_hw2)
    hw2_abs = torch.std_mean(torch.abs(errors_hw2))
    print("recons error:", f"mean {recons[0].detach():.4f}, std {recons[1].detach():.4f}")
    print("recons_abs error:", f"mean {recons_abs[0].detach():.4f}, std {recons_abs[1].detach():.4f}")
    print("hw2 error:", f"mean {hw2[0].detach():.4f}, std {hw2[1].detach():.4f}")
    print("hw2_abs error:", f"mean {hw2_abs[0].detach():.4f}, std {hw2_abs[1].detach():.4f}")
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
    print("\nIN-DISTRIBUTION VALIDATION:")
    validation_dataset_indist = TransformationDataSequence(args.validation_dataset_indist, image_size=(model.input_shape[::-1]),
                                                transform=Compose([ToTensor()]))
    print(f"{args.validation_dataset_indist} ({validation_dataset_indist.get_total_samples()})")
    validation(model, validation_dataset_indist, device, batch=100)
    validation_viz(model, validation_dataset_indist, device, batch=100, shuffle=False, sample=False, dataset_type="indist")
    viz_biggest_hw2_errors(model, validation_dataset_indist, device, batch=100, dataset_type="indist")

    # disjoint distribution validation
    print("\nDISJOINT DISTRIBUTION VALIDATION:")
    validation_dataset_disj = TransformationDataSequence(args.validation_dataset_disjoint, image_size=(model.input_shape[::-1]),
                                                  transform=Compose([ToTensor()]))
    print(f"{args.validation_dataset_disjoint} ({validation_dataset_disj.get_total_samples()})")
    validation(model, validation_dataset_disj, device, batch=100)
    validation_viz(model, validation_dataset_disj, device, batch=100, shuffle=False, sample=True, dataset_type="disjoint")


if __name__ == "__main__":
    main()