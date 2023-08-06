import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToPILImage, ToTensor
from scipy.stats import truncnorm
import cv2

# based on https://github.com/navoshta/behavioral-cloning
# adds pooling layers between each convolutional layer
class DAVE2v3(nn.Module):
    def __init__(self, input_shape=(100, 100)):
        super().__init__()
        self.input_shape = input_shape

        self.conv1 = nn.Conv2d(3, 16, 3, stride=3)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), padding=1)
        # torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=3)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        # self.conv3 = nn.Conv2d(32, 64, 3, stride=3)
        # self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        print(f"{self.input_shape=}")
        size = np.product(nn.Sequential(self.conv1, self.pool1, self.conv2, self.pool2)( #, self.conv3, self.pool3)(
            torch.zeros(1, 3, *self.input_shape)).flatten(1).shape)
        print(f"{size=}")
        self.lin1 = nn.Linear(in_features=size, out_features=500, bias=True)
        self.dropout1 = nn.Dropout(p=0.5)
        self.lin2 = nn.Linear(in_features=500, out_features=100, bias=True)
        self.dropout2 = nn.Dropout(p=0.25)
        self.lin3 = nn.Linear(in_features=100, out_features=20, bias=True)
        self.lin4 = nn.Linear(in_features=20, out_features=1, bias=True)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        # x = self.conv3(x)
        # x = F.relu(x)
        # x = self.pool3(x)
        x = x.flatten(1)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.lin3(x)
        x = F.relu(x)
        x = self.lin4(x)
        return x

    def load(self, path="test-model.pt", map_location=None):
        if map_location == None:
            map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.load(path, map_location=map_location)

    def process_image(self, image, transform=Compose([ToTensor()])):
        image = transform(np.asarray(image))[None]
        return image

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

    def load(self, path="test-model.pt", map_location=None):
        if map_location == None:
            map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.load(path, map_location=map_location)

    def process_image(self, image, transform=Compose([ToTensor()])):
        image = transform(np.asarray(image))[None]
        return image

class DAVE2extra(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = (150, 200)

        self.lr = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

        self.conv1 = nn.Conv2d(3, 32, 5, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv4 = nn.Conv2d(32, 32, 4, stride=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(0.5)

        self.lin1 = nn.Linear(8320, 512)
        self.lin2 = nn.Linear(512, 1)

        self.max_action = 1.0

    def forward(self, x):
        x = self.bn1(self.lr(self.conv1(x)))
        x = self.bn2(self.lr(self.conv2(x)))
        x = self.bn3(self.lr(self.conv3(x)))
        x = self.bn4(self.lr(self.conv4(x)))
        x = x.view(x.size(0), -1)
        # print(x.shape)# flatten
        x = self.dropout(x)
        # print(x.shape)
        x = self.lr(self.lin1(x))

        # because we don't want our duckie to go backwards
        x = self.lin2(x)
        # x[:, 0] = self.max_action * self.sigm(x[:, 0])  # because we don't want the duckie to go backwards
        # x[:, 1] = self.tanh(x[:, 1])
        x = torch.tanh(x)
        return x

    def load(self, path="test-model.pt", map_location=None):
        if map_location == None:
            map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.load(path, map_location=map_location)
