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
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kernel = 5
stride = 5
conv_stack = nn.Sequential(
                nn.Conv2d(3, 3, kernel_size=kernel,
                        stride=stride, padding=0),
                nn.ReLU(),
            )
x = torch.rand((1,3,100,100))
y = conv_stack(x)
print(f"{y.shape=}")

# kernel = 5 stride = 1 y.shape=torch.Size([1, 3, 100, 100])
# kernel = 5 stride = 5 y.shape=torch.Size([1, 3, 20, 20])
# 