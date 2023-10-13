import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import utils
from vqvae.vqvae import VQVAE
import string
import sys
import random 
from matplotlib import pyplot as plt
import os
sys.path.append("../DAVE2")
from DAVE2pytorch import DAVE2v3
import torchvision.transforms as transforms
from sklearn.metrics import mean_squared_error
from multiprocessing import Pool
import tqdm

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
parser.add_argument("--n_updates", type=int, default=5000)

parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=32)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--n_embeddings", type=int, default=512)
parser.add_argument("--beta", type=float, default=.25)

parser.add_argument("--log_interval", type=int, default=250)
parser.add_argument("--dataset",  type=str, default='CIFAR10')
parser.add_argument("--validation",  type=str, default='/p/sdbb/supervised-transformation-validation')
parser.add_argument("--filename",  type=str, default=timestamp)
parser.add_argument("--weights",  type=str, default="/p/sdbb/vqvae/results/vqvae_data_tue_may_2_16_27_22_2023.pth")
parser.add_argument("--topo",  type=str, default=None)
parser.add_argument("--basemodel",  type=str, default=None)
parser.add_argument("--transf",  type=str, default="fisheye")
parser.add_argument("--id",  type=str, default=None)

args = parser.parse_args()
print(f"{args=}", flush=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
randstr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
save_path = "TESTPARALLEL222" + args.weights.replace(".pth", "_" + args.id + "_" + randstr)
os.makedirs(save_path, exist_ok=True)
print('Results will be saved in ' + save_path)

"""
Load data and define batch data loaders
"""
training_data, validation_data, training_loader, validation_loader, x_train_var = utils.load_data_and_data_loaders(
    args.dataset, args.batch_size, shuffle=False, topo=args.topo, transf=args.transf)
"""
Set up VQ-VAE model with components defined in ./vqvae/ folder
"""
model = VQVAE(args.n_hiddens, args.n_residual_hiddens, args.n_residual_layers, args.n_embeddings, args.embedding_dim, args.beta, transf=args.transf).to(device)
checkpoint = torch.load(args.weights, map_location=device)
model.load_state_dict(checkpoint["model"])
model = model.eval()
# print(f"{checkpoint['results']=}\n\n{checkpoint['hyperparameters']=}")
basemodel = torch.load(args.basemodel, map_location=device).eval()
transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((108,192)), transforms.ToTensor()])

def calc_prediction_loss(x, x_hat):
    prediction_errors = np.zeros(x.shape[0])
    # loss = nn.MSELoss()
    for i, (xi, xhati) in enumerate(zip(x, x_hat)):
        x_i = transform(xi).to(device)
        x_hat_i = transform(xhati).to(device)
        hw1_pred = basemodel(x_i[None]).item()
        recons_pred = basemodel(x_hat_i[None]).item()
        prediction_errors[i] = mean_squared_error([hw1_pred], [recons_pred])
    return prediction_errors

def save_validation(x, x_transf, x_hat, batch=0, index=0): #, sample=None):
    print(f"{x_transf.shape=}")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, layout='constrained', sharey=True)
    x_i = transform(x[index]).to(device)
    # print(f"{x_i.shape=}")
    x_hat_i = transform(x_hat[index]).to(device)
    # print(f"{x_hat_i.shape=}")
    hw1_pred = basemodel(x_i[None]).item()
    recons_pred = basemodel(x_hat_i[None]).item()

    hw1_img = np.transpose(x[index].detach().cpu().numpy(), (1,2,0))
    recons_img = np.transpose(x_hat[index].detach().cpu().numpy(), (1,2,0))
    hw2_img = np.transpose(x_transf[index].detach().cpu().numpy(), (1,2,0))
    ax1.imshow((hw1_img * 255).astype(np.uint8))
    ax2.imshow((hw2_img * 255).astype(np.uint8))
    ax3.imshow((recons_img * 255).astype(np.uint8))

    # https://github.com/matplotlib/matplotlib/issues/1789/
    # asp = np.diff(ax1.get_xlim())[0] / np.diff(ax1.get_ylim())[0]
    # asp /= np.abs(np.diff(ax1.get_xlim())[0] / np.diff(ax1.get_ylim())[0])
    # ax2.set_aspect(asp)
    # ax1.set_aspect(asp)
    # ax2.set_aspect(asp)
    # ax3.set_aspect(asp)

    ax1.set_title(f'hw1 pred: {hw1_pred:.5f}')
    ax2.set_title(f'hw2 pred: {hw1_pred:.5f}')
    ax3.set_title(f'recons pred: {recons_pred:.5f}')

    # img_name = sample['img_name'][index].replace('/', '/\n').replace('\\', '/\n')
    maintitle = "test" #img_name.replace("None", "\n")
    fig.suptitle(f"{maintitle}\n\nHW1-recons Prediction error: {(hw1_pred - recons_pred):.5f}", fontsize=12)
    plt.savefig(f"{save_path}/output-batch{batch:04d}-sample{index:04d}.png")
    plt.close(fig)


results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
    'perplexities': [],
    'predictions': [],
}

def validateOLD(sample):
    if args.transf == "resdec" or args.transf == "resinc": # (67, 120) (270, 480)
        x_hat_transform = transforms.Resize((108, 192), antialias=True)
    else:
        x_hat_transform = None
    
    x = sample["image_base"].float().to(device)
    x_transf = sample["image_transf"].float().to(device)
    embedding_loss, x_hat, perplexity = model(x_transf)
    if x_hat_transform is not None:
        # x_base = x_hat_transform(x_hat)
        x_hat_transform = transforms.Resize((x_hat.shape[2], x_hat.shape[3]), antialias=True)
        x = x_hat_transform(x)
    prediction_loss = calc_prediction_loss(x, x_hat)
    recon_loss = torch.mean((x_hat - x)**2) / x_train_var
    loss = recon_loss + embedding_loss
    
    results["recon_errors"].append(recon_loss.cpu().detach().numpy())
    results["perplexities"].append(perplexity.cpu().detach().numpy())
    results["loss_vals"].append(loss.cpu().detach().numpy())
    results["n_updates"] = i
    results["predictions"].extend(prediction_loss)

    if i % args.log_interval == 0:
        save_validation(x, x_transf, x_hat, batch=i, sample=sample)

    print("prediction loss:", sum(results["predictions"]) / len(results["predictions"]), flush=True)

    return tuple(results["predictions"])


def validate(image_base, image_transf, i):
    if args.transf == "resdec" or args.transf == "resinc": # (67, 120) (270, 480)
        x_hat_transform = transforms.Resize((108, 192), antialias=True)
    else:
        x_hat_transform = None
    
    x = image_base.float().to(device)
    x_transf = image_transf.float().to(device)
    embedding_loss, x_hat, perplexity = model(x_transf)
    if x_hat_transform is not None:
        # x_base = x_hat_transform(x_hat)
        x_hat_transform = transforms.Resize((x_hat.shape[2], x_hat.shape[3]), antialias=True)
        x = x_hat_transform(x)
    prediction_loss = calc_prediction_loss(x, x_hat)
    recon_loss = torch.mean((x_hat - x)**2) / x_train_var
    loss = recon_loss + embedding_loss
    
    results["recon_errors"].append(recon_loss.cpu().detach().numpy())
    results["perplexities"].append(perplexity.cpu().detach().numpy())
    results["loss_vals"].append(loss.cpu().detach().numpy())
    # results["n_updates"] = i
    results["predictions"].extend(prediction_loss)

    if i % args.log_interval == 0:
        save_validation(x, x_transf, x_hat, batch=i) #, sample=sample)

    print("prediction loss:", sum(results["predictions"]) / len(results["predictions"]), flush=True)

    return tuple(results["predictions"])


def main():
    NUM_THREADS = 2
    with Pool(processes=NUM_THREADS) as pool:
        results_dict = {}
        for i in range(args.n_updates):
            print(f"Update {i}")
            sample = next(iter(validation_loader))
            # print(f"{sample.keys()=}")
            # print(f"{sample['img_name']=}")
            # print(f"{type(sample['img_name'])=}")
            # results_dict[tuple(sample["img_name"])] = pool.apply_async(validate, (sample))
            result = pool.apply_async(validate, (sample["image_base"], sample['image_transf'], i))
            # once submitted, all are running
            # for tuple(sample["img_name"]), result in tqdm.tqdm(results_dict.items()):
            return_value = result.get()  # this will wait until the result is ready
            # TODO do whatever you want with the return value
            # print(f"{return_value=}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    main()
