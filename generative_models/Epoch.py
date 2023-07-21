import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToPILImage, ToTensor
from scipy.stats import truncnorm

# based on https://github.com/udacity/self-driving-car/tree/master/steering-models/community-models/cg23
class Epoch(nn.Module):
    def __init__(self, input_shape=(100, 100)):
        super().__init__()
        self.input_shape = input_shape

        # self.conv1 = nn.Conv2d(3, 32, 3, stride=3, padding='same')
        # self.conv2 = nn.Conv2d(32, 64, 3, stride=3, padding='same')
        # self.conv3 = nn.Conv2d(64, 128, 3, stride=3, padding='same')
        self.conv1 = nn.Conv2d(3, 32, 3, padding='same')
        self.conv2 = nn.Conv2d(32, 64, 3, padding='same')
        self.conv3 = nn.Conv2d(64, 128, 3, padding='same')
        self.dropout1 = nn.Dropout(p=0.25)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        size = np.product(nn.Sequential(self.conv1, self.maxpool1, self.conv2, self.maxpool1, self.conv3, self.maxpool1 )(
            torch.zeros(1, 3, *self.input_shape)).shape)

        self.lin1 = nn.Linear(in_features=size, out_features=1024, bias=True)
        self.lin2 = nn.Linear(in_features=1024, out_features=1, bias=True)
        self.dropout2 = nn.Dropout(p=0.5)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = x.flatten(1)

        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.lin2(x)

        return x

    def load(self, path="test-model.pt"):
        return torch.load(path)

    ''' process PIL image to Tensor '''
    def process_image(self, image, transform=Compose([ToTensor()])):
        image = transform(np.asarray(image))[None]
        return image