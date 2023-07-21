import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as T

import argparse
import utils
import sys, os
# sys.path.append(f"{os.getcwd()}")
sys.path.append(f"{os.getcwd()}/../DAVE2")
from DAVE2pytorch import *
from models.vqvae import VQVAE

# from pynvml import *
# nvmlInit()
# h = nvmlDeviceGetHandleByIndex(0)
# info = nvmlDeviceGetMemoryInfo(h)
# print(f'total    : {info.total}')
# print(f'free     : {info.free}')
# print(f'used     : {info.used}')

parser = argparse.ArgumentParser()

"""
Hyperparameters
"""
timestamp = utils.readable_timestamp()

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=5000)
parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=32)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--n_embeddings", type=int, default=512)
parser.add_argument("--beta", type=float, default=.25)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--dataset",  type=str, default='CIFAR10')
parser.add_argument("--topo",  type=str, default=None)
parser.add_argument("--transf",  type=str, default="fisheye")
parser.add_argument("--basemodel",  type=str, default=None)
parser.add_argument("--max_dataset_size",  type=int, default=None)

# whether or not to save model
parser.add_argument("-save", action="store_true")
parser.add_argument("--filename",  type=str, default=timestamp)

args = parser.parse_args()
print("args:" + str(args))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("results", exist_ok=True)
if args.save:
    print('Results will be saved in ./results/vqvae_' + args.filename + "_" + timestamp + '.pth')

"""
Load data and define batch data loaders
"""

training_data, validation_data, training_loader, validation_loader, x_train_var = utils.load_data_and_data_loaders(
    args.dataset, args.batch_size, topo=args.topo, max_dataset_size=args.max_dataset_size, transf=args.transf)
"""
Set up VQ-VAE model with components defined in ./models/ folder
"""
model = VQVAE(args.n_hiddens, args.n_residual_hiddens,
              args.n_residual_layers, args.n_embeddings, args.embedding_dim, args.beta, transf=args.transf).to(device)

"""
Set up optimizer and training loop
"""
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

model.train()

results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
    'perplexities': [],
    'pred_errors': [],
}
if args.basemodel is None:
    basemodelpath = "/p/sdbb/DAVE2-Keras/DAVE2v3-108x192-82samples-5000epoch-5116933-4_12-23_11-2D8U63/model-DAVE2v3-randomblurnoise-108x192-lr1e4-5000epoch-64batch-lossMSE-82Ksamples-best152.pt"
else:
    basemodelpath = args.basemodel
print(f"{basemodelpath=}")
basemodel = torch.load(basemodelpath, map_location=device)
mseloss = torch.nn.MSELoss()
basemodel = basemodel.eval()

def prediction_loss(recon_x, x, nograd=False):
    # recon_x_resize = torchvision.transforms.Resize((135, 240))(recon_x)
    # x_resize = torchvision.transforms.Resize((135, 240))(x)
    x_base_transform = T.Resize((108, 192))
    recon_x = x_base_transform(recon_x)
    x = x_base_transform(x)
    if nograd:
        with torch.no_grad():
            pred_recons_x = basemodel(recon_x)
            pred_x = basemodel(x)
    else:
        pred_recons_x = basemodel(recon_x)
        pred_x = basemodel(x)
    return mseloss(pred_x, pred_recons_x)

def train():
    if args.transf == "resdec" or args.transf == "resinc":
        # x_base_transform = T.Resize((67, 120))
        # x_base_transform = T.Resize((270, 480))
        x_hat_transform = T.Resize((108, 192))
    else:
        x_hat_transform = None
    for i in range(args.epochs):
        for batch_idx, (x) in enumerate(training_loader):
            # x = next(iter(training_loader))
            x_base = x["image_base"].float().to(device)
            x_transf = x["image_transf"].float().to(device)
            optimizer.zero_grad()
            embedding_loss, x_hat, perplexity = model(x_transf)
            # print(f"{x_hat.shape=} {x_base.shape=} {x_transf.shape=} {x_hat.shape != x_base.shape}")
            if x_hat_transform is not None:
                # x_base = x_hat_transform(x_hat)
                x_hat_transform = T.Resize((x_hat.shape[2], x_hat.shape[3]))
                x_base = x_hat_transform(x_base)
            # print(f"{x_hat.shape=} {x_base.shape=} {x_hat.shape != x_base.shape}")
            recon_loss = torch.mean((x_hat - x_base)**2) / x_train_var
            pred_loss = prediction_loss(x_hat, x_base, nograd=False)
            loss = recon_loss + embedding_loss + 0.05 * pred_loss

            loss.backward()
            optimizer.step()

            results["recon_errors"].append(recon_loss.cpu().detach().numpy())
            results["perplexities"].append(perplexity.cpu().detach().numpy())
            results["loss_vals"].append(loss.cpu().detach().numpy())
            results["pred_errors"].append(pred_loss.cpu().detach().numpy())
            results["n_updates"] += 1

            if batch_idx % args.log_interval == 0:
                """
                save model and print values
                """
                if args.save:
                    hyperparameters = args.__dict__
                    utils.save_model_and_results(
                        model, results, hyperparameters, args.filename + "_" + timestamp)

                print('Epoch', i, 'batch', batch_idx, 'Recon Error:',
                    np.mean(results["recon_errors"][-args.log_interval:]),
                    'Loss', np.mean(results["loss_vals"][-args.log_interval:]),
                    'Perplexity:', np.mean(results["perplexities"][-args.log_interval:]), 
                    'Prediction loss:', np.mean(results["pred_errors"][-args.log_interval:]), 
                    flush=True)


if __name__ == "__main__":
    train()