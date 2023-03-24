import shutil
import random
import string
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

# meriels dependencies
from DatasetGenerator import TransformationDataSequence
from loss_functions import *
from DAVE2pytorch import *

randstr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
localtime = time.localtime()
timestr = "{}_{}-{}_{}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)
NAME = f"Lenscoder-featuresVIF-{timestr}"
Path("models").mkdir(exist_ok=True, parents=True)
Path(f"samples_{NAME}").mkdir(exist_ok=True, parents=True)
Path(f"samples_{NAME}/iter").mkdir(exist_ok=True, parents=True)
Path(f"samples_{NAME}/epoch").mkdir(exist_ok=True, parents=True)
shutil.copy(__file__, NAME)

def loss_fn_orig(recons, x, mu, log_var, kld_weight):
    recons_loss = F.mse_loss(recons, x)
    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
    )
    loss = recons_loss + kld_weight * kld_loss
    return loss, recons_loss, kld_loss


# from vif_utils import vif
from skimage.metrics import structural_similarity as ssim
from sewar.full_ref import vifp

base_model = torch.load("../weights/model-DAVE2v3-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt")
def loss_fn_features(recons, x, mu, log_var, kld_weight):
    # recons_loss = vif(recons, x)
    # two identical images have SSIM score of 1
    recons_ssim = recons.clone().detach().cpu().numpy() #.transpose()
    x_ssim = x.clone().detach().cpu().numpy()
    recons_loss = 0
    for i in range(recons_ssim.shape[0]):
        # recons_loss += (1 - ssim(recons_ssim[i], x_ssim[i],
        #               data_range=x_ssim[i].max() - x_ssim[i].min(), win_size=3)) / recons_ssim.shape[0]
        recons_loss += (1 - vifp(recons_ssim[i].transpose(1,2,0), x_ssim[i].transpose(1,2,0))) / recons_ssim.shape[0]
    pred_recons = base_model(recons)
    pred_orig = base_model(x)
    prediction_loss = F.mse_loss(pred_orig, pred_recons)
    # print(f"{prediction_loss=:.4f}   \t{recons_loss=:.4f}")
    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
    )
    loss = prediction_loss + recons_loss + kld_weight * kld_loss
    return loss, recons_loss, prediction_loss



def loss_fn(recons, x, mu, log_var, kld_weight):
    recons_loss = F.mse_loss(recons, x)
    pred_recons = base_model(recons)
    pred_orig = base_model(x)
    prediction_loss = F.mse_loss(pred_orig, pred_recons)
    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
    )
    loss = prediction_loss + recons_loss + kld_weight * kld_loss
    print(f"{prediction_loss=:.4f}\t{recons_loss=:.4f}\t{(kld_weight * kld_loss)=:.4f}")
    return loss, recons_loss, kld_loss

from skimage.metrics import structural_similarity as ssim
def train(model, data_loader, num_epochs=300, device=torch.device("cpu"), sample_interval=200, lr=0.0005):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    # TODO: adjust gamma if it plateaus at some values
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    kld_weight = 0.01 * data_loader.batch_size / len(data_loader.dataset)
    #kld_weight = data_loader.batch_size / len(data_loader.dataset)
    all_losses = []
    start_t = time.time()
    z = torch.randn(100, model.latent_dim).to(device)
    vifloss = VIFLoss(device)
    #model = model.eval()
    # grid = make_grid(model.decode(z).detach().cpu(), 10)
    # save_image(grid, f"samples_{NAME}/epoch/0.png")
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
            # recons_loss = torch.zeros(1).to(device)

            loss, rloss, kloss = vifloss(recons, y) # loss_fn_features(recons, y, mu, log_var, kld_weight) #
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            if (i % 100) == 0:
                iter_end_t = time.time()
                # epoch, samples, epoch elapsed time, latest loss, averaged 10 losses, loss term 2, loss term 3
                print(
                    f"{epoch} {i} [{iter_end_t - epoch_start_t:.1f}]: {losses[-1]:.4f} {np.mean(losses[-10:]):.4f} {rloss.item():.4f} {kloss.item():.4f}"
                )
            batches_done = (epoch - 1) * len(data_loader) + i

        epoch_end_t = time.time()
        # print(f"epoch time: {(epoch_end_t - epoch_start_t):.3f} seconds")
        # print(f"total time: {(epoch_end_t - start_t):.3f} seconds")
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
    print(f"total time: {end_t - start_t} seconds")
    return model

def validation(model, dataset, device="cpu", batch=100):
    model = model.to(device).eval()
    trainloader = DataLoader(dataset, batch_size=batch, shuffle=True)
    transform = Compose([ToTensor()])
    # z = torch.randn(100, model.latent_dim).to(device)
    for i, hashmap in enumerate(trainloader, start=1):
        x = hashmap['image_transf'].float().to(device)
        #x = transform(x)
        # y = hashmap['image_base'].float().to(device)
        recons, x_out, mu, log_var = model(x)
        grid_in = make_grid(x, 10)
        grid_out = make_grid(recons, 10)
        save_image(grid_in, f"samples_{NAME}/validation-input{i:04d}.png")
        save_image(grid_out, f"samples_{NAME}/validation-output{i:04d}.png")
        if i > 20:
            return

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='parent directory of training dataset')
    args = parser.parse_args()
    return args

def main():
    from models.VAEsteer import Model
    args = parse_arguments()
    print(args)
    start_time = time.time()
    BATCH_SIZE = 32
    NB_EPOCH = 1000
    lr = 1e-4
    robustification = False
    noise_level = 20
    latent_dim = 512
    model = Model(input_shape=(135,240), latent_dim=latent_dim)
    dataset = TransformationDataSequence(args.dataset, image_size=(model.input_shape[::-1]), transform=Compose([ToTensor()]),\
                                         robustification=robustification, noise_level=noise_level)

    print("Retrieving output distribution....")
    print("Moments of distribution:", dataset.get_outputs_distribution())
    print("Total samples:", dataset.get_total_samples())
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    trainloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=worker_init_fn)
    print("time to load dataset: {}".format(time.time() - start_time))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)

    model = model.to(device)
    model = train(model, trainloader, device=device, num_epochs=NB_EPOCH, sample_interval=20000, lr=lr)
    #model = torch.load("C:/Users/Meriel/Documents/GitHub/supervised-universal-transformation/training/samples_Lenscoder_new2old_13ksamples_1000epoch_32batch_512latent/Lenscoder_512lat.pt")
    validation(model, dataset, device, batch=100)
    model = model.to(torch.device("cpu"))
    model = model.eval()
    model_filename = f"Lenscoder_featureloss_{int(len(dataset)/1000)}k-{latent_dim}lat_{NB_EPOCH}epochs_{BATCH_SIZE}batch_{robustification}rob.pt"
    torch.save(model, model_filename)


if __name__ == "__main__":
    main()