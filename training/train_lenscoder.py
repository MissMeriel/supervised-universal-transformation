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

# meriels dependencies
from DatasetGenerator import TransformationDataSequence


NAME = "Lenscoder"
Path("models").mkdir(exist_ok=True, parents=True)
Path(f"samples_{NAME}").mkdir(exist_ok=True, parents=True)
Path(f"samples_{NAME}/iter").mkdir(exist_ok=True, parents=True)
Path(f"samples_{NAME}/epoch").mkdir(exist_ok=True, parents=True)

def loss_fn(recons, x, mu, log_var, kld_weight):
    recons_loss = F.mse_loss(recons, x)
    # recons_loss = F.binary_cross_entropy(recons, x)
    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
    )
    loss = recons_loss + kld_weight * kld_loss
    return loss, recons_loss, kld_loss


def train(model, data_loader, num_epochs=300, device=torch.device("cpu"), sample_interval=200, lr=0.0005):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    # TODO: adjust gamma if it plateaus at some values
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    # kld_weight = 0.01 * data_loader.batch_size / len(data_loader.dataset)
    kld_weight = data_loader.batch_size / len(data_loader.dataset)
    all_losses = []
    start_t = time.time()
    z = torch.randn(100, model.latent_dim).to(device)
    model = model.eval()
    # grid = make_grid(model.decode(z).detach().cpu(), 10)
    # save_image(grid, f"samples_{NAME}/epoch/0.png")
    for epoch in range(1, num_epochs + 1):
        epoch_start_t = time.time()
        losses = []
        model = model.train()
        for i, hashmap in enumerate(data_loader, start=1):
            optimizer.zero_grad()
            x = hashmap['image_base'].float().to(device)
            # y = hashmap['steering_input'].float().to(device)
            y = hashmap['image_transf'].float().to(device)
            x = Variable(x, requires_grad=True)
            y = Variable(y, requires_grad=False)
            recons, x, mu, log_var = model(x)
            loss, rloss, kloss = loss_fn(recons, y, mu, log_var, kld_weight)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            if (i % 100) == 0:
                iter_end_t = time.time()
                # epoch, samples, epoch elapsed time,
                print(
                    f"{epoch} {i} [{iter_end_t - epoch_start_t:.1f}]: {losses[-1]:.4f} {np.mean(losses[-10:]):.4f} {rloss.item():.4f} {kloss.item():.4f}"
                )
            batches_done = (epoch - 1) * len(data_loader) + i

        epoch_end_t = time.time()
        print(f"epoch time: {epoch_end_t - epoch_start_t} seconds")
        print(f"total time: {epoch_end_t - start_t} seconds")
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
    BATCH_SIZE = 64
    NB_EPOCH = 100
    lr = 1e-4
    robustification = False
    noise_level = 20
    model = Model(input_shape=(135,240), latent_dim=512)
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
    model = model.to(torch.device("cpu"))
    model = model.eval()
    model_filename = "Lenscoder.pt"
    torch.save(model, model_filename)


if __name__ == "__main__":
    main()