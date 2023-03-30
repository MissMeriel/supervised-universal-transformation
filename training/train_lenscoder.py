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

# meriels dependencies
from DatasetGenerator import TransformationDataSequence
# from VIF_loss_functions import *
# from SSIM_loss_functions import *

sys.path.append(f'../models')
from basic_loss_functions import *
from models.DAVE2pytorch import *

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", '--training_dataset', help='parent directory of training dataset')
    parser.add_argument('-v', '--validation_dataset', help='parent directory of training dataset')
    parser.add_argument('-o', '--procid', type=int, help='identifier or slurm process id')
    parser.add_argument('-e', '--epochs', type=int, default=1000, help='training epochs')
    parser.add_argument('-l', '--loss_fn', type=str, default="orig", help='loss fn type')
    parser.add_argument('-r', '--lr', type=float, default=0.0001, help='learning rate')
    args = parser.parse_args()
    return args

args = parse_arguments()
randstr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
localtime = time.localtime()
timestr = "{}_{}-{}_{}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)
NAME = f"samples_Lenscoder-latentloss-3_28-11_47-Lenscoder-{args.loss_fn}loss-dataindist-{args.procid}-{timestr}-{randstr}"
Path("models").mkdir(exist_ok=True, parents=True)
Path(f"samples_{NAME}").mkdir(exist_ok=True, parents=True)
Path(f"samples_{NAME}/iter").mkdir(exist_ok=True, parents=True)
Path(f"samples_{NAME}/epoch").mkdir(exist_ok=True, parents=True)
shutil.copy(__file__, f"samples_{NAME}")
shutil.copy("basic_loss_functions.py", f"samples_{NAME}")


def train(model, data_loader, num_epochs=300, device=torch.device("cpu"), sample_interval=200, lr=0.0005):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # TODO: adjust gamma if it plateaus at some values
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    kld_weight = 0.01 * data_loader.batch_size / len(data_loader.dataset)
    #kld_weight = data_loader.batch_size / len(data_loader.dataset)
    all_losses = []
    start_t = time.time()
    z = torch.randn(100, model.latent_dim).to(device)
    for epoch in range(1, num_epochs + 1):
        epoch_start_t = time.time()
        losses = []
        model = model.train()
        for i, hashmap in enumerate(data_loader, start=1):
            optimizer.zero_grad()
            x = hashmap['image_transf'].float().to(device)
            # y = hashmap['steering_input'].float().to(device)
            y = hashmap['image_base'].float().to(device)
            x = Variable(x, requires_grad=True)
            y = Variable(y, requires_grad=False)
            recons, x, mu, log_var = model(x)
            if args.loss_fn == "orig":
                loss, rloss, kloss = loss_fn_orig(recons, y, mu, log_var, kld_weight)
            elif args.loss_fn == "latent_kld":
                loss, rloss, kloss = loss_fn_latent_kld(recons, y, mu, log_var, kld_weight)
            elif args.loss_fn == "features_kld":
                loss, rloss, kloss = loss_fn_features_kld(recons, y, mu, log_var, kld_weight)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            if (i % 100) == 0:
                iter_end_t = time.time()
                # epoch, samples, epoch elapsed time, latest loss, averaged 10 losses, loss term 2, loss term 3
                print(
                    f"{epoch} {i} [{iter_end_t - epoch_start_t:.1f}]: {losses[-1]:.4f} {np.mean(losses[-10:]):.4f} {rloss.item():.4f} {kloss.item():.4f}",
                    flush=True
                )
            batches_done = (epoch - 1) * len(data_loader) + i

        epoch_end_t = time.time()
        all_losses.append(losses)
        lr_scheduler.step()
        model = model.eval()
        save_image(
            model.sample(25).detach().cpu(),
            f"samples_{NAME}/iter/%d.png" % batches_done,
            nrow=5,
        )
        grid = make_grid(model.decode(z).detach().cpu(), 10)
        save_image(grid, f"samples_{NAME}/epoch/{epoch}.png")
    end_t = time.time()
    print(f"total time: {end_t - start_t} seconds", flush=True)
    return model

# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/figure_title.html
def validation(vae, dataset, device="cpu", batch=100):
    import threading
    plot_lock = threading.Lock()


    vae = vae.to(device).eval()
    base_model = torch.load(
        "../weights/model-DAVE2v3-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt").to(device)
    trainloader = DataLoader(dataset, batch_size=batch, shuffle=False)
    transform = Compose([ToTensor()])
    Path(f"samples_{NAME}/validation-sets").mkdir(exist_ok=True, parents=True)
    Path(f"samples_{NAME}/validation-indv").mkdir(exist_ok=True, parents=True)
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
                    fig.suptitle(f"{img_name}\n\nHW1-HW2 Prediction error: {(pred_hw1[j][0] - pred_hw2[j][0]):.5f}\n\nHW1-recons Prediction error: {(pred_hw1[j][0] - pred_recons[j][0]):.5f}", fontsize=12)
                    # fig.suptitle(f"{hashmap['img_name']}\nPrediction error: {(pred_hw1[j][0] - pred_recons[j][0]):.5f}", fontsize=16)
                    plt.savefig(f"samples_{NAME}/validation-indv/output-batch{i:04d}-sample{j:04d}.png")
                    plt.close(fig)
            if i > 5:
                return


# from torch.utils.tensorboard import SummaryWriter
# def tb_summary(x, recons, pred_orig, pred_recons):
#
#     pass


def main():
    from models.VAEsteer import Model
    print(args, flush=True)
    start_time = time.time()
    BATCH_SIZE = 32
    NB_EPOCH = args.epochs
    lr = args.lr
    robustification = False
    noise_level = 20
    latent_dim = 512
    model = Model(input_shape=(135,240), latent_dim=latent_dim)
    training_dataset = TransformationDataSequence(args.training_dataset, image_size=(model.input_shape[::-1]), transform=Compose([ToTensor()]),\
                                         robustification=robustification, noise_level=noise_level)

    print("Moments of distribution:", training_dataset.get_outputs_distribution(), flush=True)
    print("Total samples:", training_dataset.get_total_samples(), flush=True)
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    trainloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=worker_init_fn)
    print(f"time to load dataset: {(time.time() - start_time):.3f}", flush=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device, flush=True)

    # model = model.to(device)
    # model = train(model, trainloader, device=device, num_epochs=NB_EPOCH, sample_interval=20000, lr=lr)
    # model_filename = f"samples_{NAME}/Lenscoder_featureloss_{int(len(training_dataset)/1000)}k-{latent_dim}lat_{NB_EPOCH}epochs_{BATCH_SIZE}batch_{robustification}rob.pt"
    # model = model.to(torch.device("cpu"))
    # model = model.eval()
    # torch.save(model, model_filename)

    # C:\Users\Meriel\Documents\GitHub\supervised-universal-transformation\training\samples_Lenscoder-convKLDloss-dataindist-3_29-14_45-NO1QFP
    model = torch.load("C:/Users/Meriel/Documents/GitHub/supervised-universal-transformation/training/samples_Lenscoder-latentloss-3_28-11_47/Lenscoder_featureloss_38k-512lat_1000epochs_32batch_Falserob.pt").to(device)
    validation_dataset = TransformationDataSequence(args.validation_dataset, image_size=(model.input_shape[::-1]),
                                                  transform=Compose([ToTensor()]), \
                                                  robustification=robustification, noise_level=noise_level)
    validation(model, validation_dataset, device, batch=100)




if __name__ == "__main__":
    main()