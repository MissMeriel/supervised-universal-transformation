import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import sys
sys.path.append('..')
import argparse
from models.vqvae import VQVAE
import sys
from PIL import Image

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
import numpy as np
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=32)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--n_embeddings", type=int, default=512)
parser.add_argument("--beta", type=float, default=.25)
# parser.add_argument("-save", action="store_true")

transf="resinc"
args = parser.parse_args()
print("args:" + str(args))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VQVAE(args.n_hiddens, args.n_residual_hiddens,
              args.n_residual_layers, args.n_embeddings, args.embedding_dim, args.beta, transf=transf).to(device)
if transf == "resinc":
    img_filename = "/p/sdbb/supervised-transformation-dataset-alltransforms3/utah-14963-extra_utahlong2-fisheye.None-run00-6_13-14_11-7ZYMOA/sample-hires-01947.jpg"
elif transf == "resdec":
    img_filename = "/p/sdbb/supervised-transformation-dataset-alltransforms3/utah-14963-extra_utahlong2-fisheye.None-run00-6_13-14_11-7ZYMOA/sample-lores-01947.jpg"
elif transf == "fisheye":
    img_filename = "/p/sdbb/supervised-transformation-dataset-alltransforms3/utah-14963-extra_utahlong2-fisheye.None-run00-6_13-14_11-7ZYMOA/sample-transf-01947.jpg"
elif transf == "depth":
    img_filename = "/p/sdbb/supervised-transformation-dataset-alltransforms3/utah-14963-extra_utahlong2-fisheye.None-run00-6_13-14_11-7ZYMOA/sample-base-01947.jpg"
    img_depth_filename = "/p/sdbb/supervised-transformation-dataset-alltransforms3/utah-14963-extra_utahlong2-fisheye.None-run00-6_13-14_11-7ZYMOA/sample-depth-01947.jpg"
img = Image.open(img_filename)
print(f"{img.size=}")
if transf == "fisheye" or transf == "depth":
    img = img.resize((192, 108))
elif transf == "resdec":
    # img = img.resize((120, 67))
    img = img.resize((96, 54))
elif transf == "resinc":
    img = img.resize((480, 270))
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
img_tensor = transform(img)
print(f"{img_tensor.shape=}")
embedding_loss, x_hat, perplexity = model(img_tensor[None], verbose=True)
print(f"{embedding_loss=}, {x_hat.shape=}, {perplexity=}")