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

sys.path.append("/p/sdbb/BBTG/VAE")

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
    parser.add_argument('-v', '--validation_dataset', default=None, help='parent directory of training dataset')
    parser.add_argument('-o', '--procid', type=int, help='identifier or slurm process id')
    parser.add_argument('-e', '--epochs', type=int, default=1000, help='training epochs')
    parser.add_argument('-l', '--loss_fn', type=str, default="orig", help='loss fn type')
    parser.add_argument('-r', '--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('-d', '--latent_dim', type=int, default=512, help='latent dimension of VAE')
    args = parser.parse_args()
    return args


args = parse_arguments()
print(args, flush=True)
randstr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
localtime = time.localtime()
timestr = "{}_{}-{}_{}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)
dataset_id = args.training_dataset.split("-")[-1]
dataset_id = dataset_id.replace("/", "").replace("\\", "")
NAME = f"{dataset_id}-Lenscoder-{args.loss_fn}loss-{args.procid}-{timestr}-{randstr}"
Path(f"samples_{NAME}").mkdir(exist_ok=True, parents=True)
Path(f"samples_{NAME}/iter").mkdir(exist_ok=True, parents=True)
Path(f"samples_{NAME}/epoch").mkdir(exist_ok=True, parents=True)
shutil.copy(__file__, f"samples_{NAME}")
shutil.copy("basic_loss_functions.py", f"samples_{NAME}")
print(f"Running {__file__}")
print(f"Logging to samples_{NAME}")


def train(model, data_loader, num_epochs=300, device=torch.device("cpu"), sample_interval=100, lr=0.0005):
    model = model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # TODO: adjust gamma if it plateaus at some values
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    kld_weight = 0.01 * data_loader.batch_size / len(data_loader.dataset)
    # kld_weight = data_loader.batch_size / len(data_loader.dataset)
    print(f"{len(data_loader.dataset)=}")
    print(f"{len(data_loader)=}")
    all_losses = []
    start_t = time.time()
    z = torch.randn(100, model.latent_dim).to(device)
    tot_recons_loss = np.zeros(num_epochs)
    tot_pred_loss = np.zeros(num_epochs) 
    tot_kld_loss = np.zeros(num_epochs)
    for epoch in range(1, num_epochs + 1):
        epoch_start_t = time.time()
        losses = np.zeros(len(data_loader))
        recons_losses = np.zeros(len(data_loader))
        pred_losses = np.zeros(len(data_loader))
        kld_losses = np.zeros(len(data_loader))
        model = model.train()
        index = 0
        for i, hashmap in enumerate(data_loader, start=1):
            optimizer.zero_grad()
            x = hashmap['image_transf'].float().to(device) #* 256
            y = hashmap['image_base'].float().to(device) #* 256
            x = Variable(x, requires_grad=True)
            y = Variable(y, requires_grad=False)
            recons, x, mu, log_var = model(x)
            if args.loss_fn == "orig":
                loss, rloss, kloss = loss_fn_orig(recons, y, mu, log_var, kld_weight)
            elif args.loss_fn == "latent_kld":
                loss, rloss, kloss = loss_fn_latent_kld(recons, y, mu, log_var, kld_weight)
            elif args.loss_fn == "features_kld":
                loss, rloss, kloss = loss_fn_features_kld(recons, y, mu, log_var, kld_weight)
            elif args.loss_fn == "prediction_kld":
                loss, rloss, kloss, kld_loss = loss_fn_prediction_kld(recons, y, mu, log_var, kld_weight)
                # print(loss, rloss, kloss, kld_loss)
                recons_losses[index] = torch.mean(rloss)
                pred_losses[index] = torch.mean(kloss)
                kld_losses[index] = torch.mean(kld_loss)

            elif args.loss_fn == "weighted_prediction_kld":
                loss, rloss, kloss, kld_loss = loss_fn_prediction_kld(recons, y, mu, log_var, kld_weight, term_weights=(1.5,2,1))
                recons_losses[index] = rloss.item()
                pred_losses[index] = kloss.item()
                kld_losses[index] = kld_loss.item()
            
            elif args.loss_fn == "fid_kld":
                loss, rloss, kloss = loss_fn_fid_kld(recons, y, mu, log_var, kld_weight)
            elif args.loss_fn == "prediction_only":
                loss, rloss, kloss = loss_fn_prediction_only(recons, y, mu, log_var, kld_weight)
            loss.backward()
            losses[index] = loss.item()
            optimizer.step()
            index += 1

            if (i % sample_interval) == 0:
                iter_end_t = time.time()
                # epoch, samples, epoch elapsed time, latest loss, averaged 10 losses, loss term 2, loss term 3
                print(
                    f"{epoch} {i} [{iter_end_t - epoch_start_t:.1f}]: {losses[-1]:.4f} {np.mean(losses[-10:]):.4f} {rloss.item():.4f} {kloss.item():.4f}",
                    flush=True
                )

            # print(f"{index=} \t {x.shape=} \t{i=}")
            
        batches_done = (epoch - 1) * len(data_loader) + i
        tot_recons_loss[epoch-1] = sum(recons_losses) / len(data_loader)
        tot_pred_loss[epoch-1] = sum(pred_losses) / len(data_loader)
        tot_kld_loss[epoch-1] = sum(kld_losses) / len(data_loader)
        # print(f"Epoch {epoch} losses by term:\n\t{tot_recons_loss[epoch-1]=} \n\t{tot_pred_loss[epoch-1]=} \n\t{tot_kld_loss[epoch-1]=}\n")
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
    print(f"Avg losses by term:")
    print(f"\t{np.mean(tot_recons_loss)=}")
    print(f"\t{np.mean(tot_pred_loss)=}")
    print(f"\t{np.mean(tot_kld_loss)=}")
    end_t = time.time()
    print(f"total time: {end_t - start_t} seconds", flush=True)
    return model


# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/figure_title.html
def validation_viz(vae, dataset, device="cpu", batch=100):
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
                    # fig.suptitle(f"{hashmap['img_name']}\nPrediction error: {(pred_hw1[j][0] - pred_recons[j][0]):.5f}", fontsize=16)
                    plt.savefig(f"samples_{NAME}/validation-indv/output-batch{i:04d}-sample{j:04d}.png")
                    plt.close(fig)
            if i > 5:
                return


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
    print("recons error:", recons)
    print("recons_abs error:", recons_abs)
    print("hw2 error:", hw2)
    print("hw2_abs error:", hw2_abs)
    return recons, hw2


def main():
    from models.VAEsteer import Model
    start_time = time.time()
    BATCH_SIZE = 32
    NB_EPOCH = args.epochs
    lr = args.lr
    robustification = False
    noise_level = 20
    latent_dim = args.latent_dim
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

    model = model.to(device)
    model = train(model, trainloader, device=device, num_epochs=NB_EPOCH, sample_interval=200, lr=lr)
    model_filename = f"samples_{NAME}/Lenscoder_{int(len(training_dataset)/1000)}k-{latent_dim}lat_{NB_EPOCH}epochs_{BATCH_SIZE}batch_{robustification}rob.pt"
    model = model.to(torch.device("cpu"))
    model = model.eval()
    torch.save(model, model_filename)
    
    if args.validation_dataset is not None:
        validation_dataset = TransformationDataSequence(args.validation_dataset, image_size=(model.input_shape[::-1]),
                                                    transform=Compose([ToTensor()]), \
                                                    robustification=robustification, noise_level=noise_level)
        validation(model, validation_dataset, device, batch=100)
        validation_viz(model, validation_dataset, device, batch=100)


if __name__ == "__main__":
    main()