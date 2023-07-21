import numpy as np
import os, cv2, csv

import kornia

from PIL import Image
import copy
from scipy import stats
# adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
import torch.utils.data as data
from pathlib import Path
import skimage.io as sio
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import random

from torchvision.transforms import Compose, ToTensor, PILToTensor, functional as transforms
from io import BytesIO
import skimage

def stripleftchars(s):
    # print(f"{s=}")
    for i in range(len(s)):
        if s[i].isnumeric():
            return s[i:]
    return -1

class DataSequence(data.Dataset):
    def __init__(self, root, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.transform = transform

        image_paths = []
        for p in Path(root).iterdir():
            if p.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"]:
                image_paths.append(p)
        image_paths.sort(key=lambda p: int(stripleftchars(p.stem)))
        self.image_paths = image_paths
        # print(f"{self.image_paths=}")
        self.df = pd.read_csv(f"{self.root}/data.csv")
        self.cache = {}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        img_name = self.image_paths[idx]
        image = sio.imread(img_name)

        df_index = self.df.index[self.df['filename'] == img_name.name]
        y_thro = self.df.loc[df_index, 'throttle_input'].array[0]
        y_steer = self.df.loc[df_index, 'steering_input'].array[0]
        y = [y_steer, y_thro]
        # torch.stack(y, dim=1)
        y = torch.tensor(y_steer)

        # plt.title(f"steering_input={y_steer.array[0]}")
        # plt.imshow(image)
        # plt.show()
        # plt.pause(0.01)

        if self.transform:
            image = self.transform(image).float()
        # print(f"{img_name.name=} {y_steer=}")
        # print(f"{image=}")
        # print(f"{type(image)=}")
        # print(self.df)
        # print(y_steer.array[0])

        # sample = {"image": image, "steering_input": y_steer.array[0]}
        sample = {"image": image, "steering_input": y}

        self.cache[idx] = sample
        return sample

'''parent dir containing multiple transformation datasets'''
class TransformationDataSequence(data.Dataset):
    def __init__(self, root, image_size=(100,100), transform=None, robustification=False, noise_level=10, key=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.transform = transform
        self.size = 0
        self.image_size = image_size[:2]
        image_paths_hashmap = {}
        all_image_paths = []
        self.dfs_hashmap = {}
        self.dirs = []
        for p in Path(root).iterdir():
            if p.is_dir() and ((key is None) or (key is not None and key in str(p))):
                self.dirs.append("{}/{}".format(p.parent, p.stem))
                image_paths = []
                try:
                    self.dfs_hashmap[f"{p}"] = pd.read_csv(f"{p}/data.txt")
                except FileNotFoundError as e:
                    try:
                        self.dfs_hashmap[f"{p}"] = pd.read_csv(f"{p}/data.csv")
                    except FileNotFoundError as e:
                        print(e, "\nNo data.txt or data.csv in directory")
                        continue
                for pp in Path(p).iterdir():
                    if pp.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"] and "transf" not in pp.name:
                        image_paths.append(pp)
                        all_image_paths.append(pp)
                image_paths.sort(key=lambda p: int(stripleftchars(p.stem)))
                image_paths_hashmap[p] = copy.deepcopy(image_paths)
                self.size += len(image_paths)
        print("Finished intaking image paths from {}! (size {})".format(self.root, self.size))
        self.image_paths_hashmap = image_paths_hashmap
        self.all_image_paths = all_image_paths
        # self.df = pd.read_csv(f"{self.root}/data.csv")
        self.cache = {}
        self.robustification = robustification
        self.noise_level = noise_level

    def get_total_samples(self):
        return self.size

    def get_directories(self):
        return self.dirs
        
    def __len__(self):
        return len(self.all_image_paths)

    def robustify(self, image_base, image_transf, y_steer):
        if random.random() > 0.75:
            image_base = torch.flip(image_base, (2,))
            image_transf = torch.flip(image_transf, (2,))
            y_steer = -y_steer
        if random.random() > 0.75:
            gauss = kornia.filters.GaussianBlur2d((3, 3), (1.5, 1.5))
            image_base = gauss(image_base[None])[0]
        if random.random() > 0.75:
            gauss = kornia.filters.GaussianBlur2d((3, 3), (1.5, 1.5))
            image_transf = gauss(image_transf[None])[0]
        if self.noise_level is not None:
            image_base = torch.clamp(image_base + (torch.randn(*image_base.shape) / self.noise_level), 0, 1)
            image_transf = torch.clamp(image_transf + (torch.randn(*image_transf.shape) / self.noise_level), 0, 1)
        return image_base, image_transf, y_steer

    def __getitem__(self, idx):
        if idx in self.cache:
            if self.robustification:
                sample = self.cache[idx]
                y_steer = sample["steering_input"]
                image_base = copy.deepcopy(sample["image_base"])
                image_transf = copy.deepcopy(sample["image_transf"])
                image_base, image_transf, y_steer = self.robustify(image_base, image_transf, y_steer)
                return {"image_base": image_base, "image_transf": image_transf, "steering_input": y_steer, "throttle_input": sample["throttle_input"],  "img_name": sample["img_name"], "all": torch.FloatTensor([y_steer, sample["throttle_input"]])}
            else:
                return self.cache[idx]
        img_name = self.all_image_paths[idx]
        image_base_orig = Image.open(img_name)
        # print(self.image_size)
        width, height = image_base_orig.size
        left = 0
        right = width
        top = 63
        bottom = height
        # image_base_orig.save("orig_test.jpg")
        # image_base_orig = image_base_orig.crop((left, top, right, bottom))
        image_base_orig = image_base_orig.resize(self.image_size)
        # image_base_orig.save("crop_test.jpg")

        img_transf_path = str(img_name).replace("base", "transf")
        image_transf_orig = Image.open(img_transf_path)
        # image_transf_orig = image_transf_orig.crop((left, top, right, bottom))
        image_transf_orig = image_transf_orig.resize(self.image_size)
        image_base_orig = self.transform(image_base_orig)
        image_transf_orig = self.transform(image_transf_orig)
        pathobj = Path(img_name)
        df = self.dfs_hashmap[f"{pathobj.parent}"]
        df_index = df.index[df['IMG'] == img_name.name]
        orig_y_steer = df.loc[df_index, 'PREDICTION'].item()
        # y_throttle = df.loc[df_index, 'throttle_input'].item()
        y_steer = copy.deepcopy(orig_y_steer)
        if self.robustification:
            image_base, image_transf, y_steer = self.robustify(copy.deepcopy(image_base_orig), copy.deepcopy(image_transf_orig), y_steer)
        else:
            image_base, image_transf = copy.deepcopy(image_base_orig), copy.deepcopy(image_transf_orig)
            # t = Compose([ToTensor()])
            # image_base_orig = t(image_base_orig).float()
            # image_transf_orig = t(image_transf_orig).float()

        sample = {"image_base": image_base, "image_transf": image_transf, "steering_input": torch.FloatTensor([y_steer]), "img_name": str(img_name), "all": torch.FloatTensor([y_steer])}
        orig_sample = {"image_base": image_base_orig, "image_transf": image_transf_orig,"steering_input": torch.FloatTensor([orig_y_steer]), "img_name": str(img_name), "all": torch.FloatTensor([orig_y_steer])}
        # try:
        #     self.cache[idx] = orig_sample
        # except MemoryError as e:
        #     print(f"Memory error adding sample to cache: {e}", flush=True)
        return sample

    def get_inputs_distribution(self):
        all_outputs = np.array([])
        for key in self.dfs_hashmap.keys():
            df = self.dfs_hashmap[key]
            # IMG,PREDICTION,POSITION,ORIENTATION,KPH,STEERING_ANGLE_CURRENT
            img_name = df['image_base']
            img_base = Image.open(img_name)
            # print("len(arr)=", len(arr))
            all_outputs = np.concatenate((all_outputs, arr), axis=0)
            # print(f"Retrieved dataframe {key=}")
        all_outputs = np.array(all_outputs)
        moments = self.get_distribution_moments(all_outputs)
        return moments
    
    def get_outputs_distribution(self):
        all_outputs = np.array([])
        for key in self.dfs_hashmap.keys():
            df = self.dfs_hashmap[key]
            # IMG,PREDICTION,POSITION,ORIENTATION,KPH,STEERING_ANGLE_CURRENT
            arr = df['PREDICTION'].to_numpy()
            # print("len(arr)=", len(arr))
            all_outputs = np.concatenate((all_outputs, arr), axis=0)
            # print(f"Retrieved dataframe {key=}")
        all_outputs = np.array(all_outputs)
        moments = self.get_distribution_moments(all_outputs)
        return moments
    
    def get_all_outputs(self):
        all_outputs = np.array([])
        for key in self.dfs_hashmap.keys():
            df = self.dfs_hashmap[key]
            # IMG,PREDICTION,POSITION,ORIENTATION,KPH,STEERING_ANGLE_CURRENT
            arr = df['PREDICTION'].to_numpy()
            # print("len(arr)=", len(arr))
            all_outputs = np.concatenate((all_outputs, arr), axis=0)
            # print(f"Retrieved dataframe {key=}")
        return np.array(all_outputs)

    ##################################################
    # ANALYSIS METHODS
    ##################################################

    # Moments are 1=mean 2=variance 3=skewness, 4=kurtosis
    def get_distribution_moments(self, arr):
        moments = {}
        moments['shape'] = np.asarray(arr).shape
        moments['mean'] = np.mean(arr)
        moments['median'] = np.median(arr)
        moments['var'] = np.var(arr)
        moments['skew'] = stats.skew(arr)
        moments['kurtosis'] = stats.kurtosis(arr)
        moments['max'] = max(arr)
        moments['min'] = min(arr)
        return moments
