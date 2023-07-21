import numpy as np
import os, cv2, csv
import math
import kornia

from PIL import Image
import copy
from scipy import stats
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
import utils

def stripleftchars(s):
    for i in range(len(s)):
        if s[i].isnumeric():
            return s[i:].split("-")[-1]
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
        self.df = pd.read_csv(f"{self.root}/data.csv")
        self.cache = {}

    def __len__(self):
        return self.size #len(self.image_paths)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        img_name = self.image_paths[idx]
        image = sio.imread(img_name)

        df_index = self.df.index[self.df['filename'] == img_name.name]
        y_thro = self.df.loc[df_index, 'throttle_input'].array[0]
        y_steer = self.df.loc[df_index, 'steering_input'].array[0]
        y = [y_steer, y_thro]
        y = torch.tensor(y_steer)

        if self.transform:
            image = self.transform(image).float()

        sample = {"image": image, "steering_input": y}

        self.cache[idx] = sample
        return sample

'''parent dir containing multiple transformation datasets'''
class TransformationDataSequence(data.Dataset):
    def __init__(self, root, image_size=(100,100), transform=None, robustification=False, noise_level=10, 
                 key=None, key2=None, max_dataset_size=None, transf=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.transform = transform
        self.transf = transf
        self.size = 0
        self.image_size = image_size[:2]
        image_paths_hashmap = {}
        all_image_paths = []
        self.dfs_hashmap = {}
        self.dirs = []
        # marker = "ep"
        for p in Path(root).iterdir():
            # if p.is_dir() and marker in str(p):
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
                # TODO: FIX FOR DS SIZE LIMIT
                for pp in Path(p).iterdir():
                    # if pp.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"] and "transf" not in pp.name:
                    if pp.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"] and "sample-base" in pp.name:
                        image_paths.append(pp)
                        all_image_paths.append(pp)
                image_paths.sort(key=lambda p: int(stripleftchars(p.stem)))
                image_paths_hashmap[p] = copy.deepcopy(image_paths)
                self.size += len(image_paths)
        print(f"Full dataset size: {self.size=}")
        print(f"{max_dataset_size=}")
        if max_dataset_size is not None:
            if self.size < max_dataset_size:
                print(f"Dataset size is {self.size} ({max_dataset_size - self.size} less than {max_dataset_size=}. Not limiting dataset size.)")
                exit(0)
            else:
                ratio = (self.size - max_dataset_size) / self.size 
                for key in self.dfs_hashmap.keys():
                    # print(f"\n{key=}")
                    # print(f"dataframe size before drop: {self.dfs_hashmap[key].shape[0]}")
                    df_size = self.dfs_hashmap[key].shape[0]
                    deductible = math.ceil(self.dfs_hashmap[key].shape[0] * ratio)
                    # print(f"{deductible=} \t{ratio=}")
                    drop_index = pd.RangeIndex(start=self.dfs_hashmap[key].index[df_size - deductible ], stop=self.dfs_hashmap[key].index[self.dfs_hashmap[key].shape[0]-1])
                    # print(f"{drop_index=}")
                    self.dfs_hashmap[key].drop(drop_index, axis=0, inplace = True)
                    #self.dfs_hashmap[key].drop(deductible, inplace = True)
                    # print(f"dataframe size after drop: {self.dfs_hashmap[key].shape[0]}", flush=True)
                dropped_dataset_size = 0
                for key in self.dfs_hashmap.keys():
                    dropped_dataset_size += self.dfs_hashmap[key].shape[0]
                # print(f"{dropped_dataset_size=}")
                self.size = dropped_dataset_size
                # print(f"{all_image_paths=}")
        all_image_paths = []
        image_paths_hashmap = {}
        for key in self.dfs_hashmap.keys():
            image_paths = []
            # print(f"{self.dfs_hashmap[key].columns=}")
            for pp in Path(key).iterdir():
                # if pp.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"] and "transf" not in pp.name:
                if pp.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"] and "sample-base" in pp.name:
                    try:
                        df_index = self.dfs_hashmap[key].index[self.dfs_hashmap[key]['IMG'] == pp.name]
                    except KeyError as e:
                        df_index = self.dfs_hashmap[key].index[self.dfs_hashmap[key]['filename'] == pp.name]
                    if not df_index.empty:
                        image_paths.append(pp)
                        all_image_paths.append(pp)
            # print(f"{pp=} \t{pp.name=}")
            image_paths.sort(key=lambda p: int(stripleftchars(p.stem)))
            image_paths_hashmap[key] = copy.deepcopy(image_paths)

        print("Finished intaking image paths from {}! (size {})".format(self.root, self.size))
        self.image_paths_hashmap = image_paths_hashmap
        self.all_image_paths = all_image_paths
        print(f"Dataset size is {len(self.all_image_paths)}")
        # self.df = pd.read_csv(f"{self.root}/data.csv")
        self.cache = {}
        self.robustification = robustification
        self.noise_level = noise_level

    def get_total_samples(self):
        return self.size

    def get_directories(self):
        return self.dirs
        
    def __len__(self):
        return self.size #len(self.all_image_paths)

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
        width, height = image_base_orig.size
        left = 0
        right = width
        top = 63
        bottom = height
        # image_base_orig.save("orig_test.jpg")
        # image_base_orig = image_base_orig.crop((left, top, right, bottom))
        image_base_orig = image_base_orig.resize(self.image_size)
        # image_base_orig.save("crop_test.jpg")
        if self.transf == "depth":
            image_transf_orig = utils.get_depth_image(img_name)
            image_transf_orig = Image.fromarray((image_transf_orig * 255).astype(np.uint8))
            # print(f"After utils.get_depth_image {type(image_transf_orig)=}")
        elif self.transf == "fisheye":
            img_transf_path = str(img_name).replace("base", "transf")
            image_transf_orig = Image.open(img_transf_path)
            # image_transf_orig = image_transf_orig.crop((left, top, right, bottom))
        elif self.transf == "resdec":
            img_transf_path = str(img_name).replace("base", "lores")
            image_transf_orig = Image.open(img_transf_path)
        elif self.transf == "resinc":
            img_transf_path = str(img_name).replace("base", "hires")
            image_transf_orig = Image.open(img_transf_path)
        image_transf_orig = image_transf_orig.resize(self.image_size)
        image_transf_orig = np.array(image_transf_orig)
        # print(f"{type(image_transf_orig)=}")
        if self.transform is not None:
            image_base_orig = self.transform(image_base_orig)
            image_transf_orig = self.transform(image_transf_orig)
            # print(f"{type(image_transf_orig)=}")
        pathobj = Path(img_name)
        df = self.dfs_hashmap[f"{pathobj.parent}"]
        try:
            df_index = df.index[df['filename'] == img_name.name]
        except KeyError as e:
            df_index = df.index[df['IMG'] == img_name.name]
        try:
            orig_y_steer = df.loc[df_index, 'steering_input'].item()
        except KeyError as e:
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
            all_outputs = np.concatenate((all_outputs, arr), axis=0)
        all_outputs = np.array(all_outputs)
        moments = self.get_distribution_moments(all_outputs)
        return moments
    
    def get_outputs_distribution(self):
        all_outputs = np.array([])
        for key in self.dfs_hashmap.keys():
            df = self.dfs_hashmap[key]
            # IMG,PREDICTION,POSITION,ORIENTATION,KPH,STEERING_ANGLE_CURRENT
            arr = df['PREDICTION'].to_numpy()
            all_outputs = np.concatenate((all_outputs, arr), axis=0)
        all_outputs = np.array(all_outputs)
        moments = self.get_distribution_moments(all_outputs)
        return moments
    
    def get_all_outputs(self):
        all_outputs = np.array([])
        for key in self.dfs_hashmap.keys():
            df = self.dfs_hashmap[key]
            arr = df['PREDICTION'].to_numpy()
            all_outputs = np.concatenate((all_outputs, arr), axis=0)
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
