import numpy as np
import sys
import os, cv2, csv
from DAVE2pytorch import DAVE2PytorchModel
import kornia
import math
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
sys.path.append(f"{os.getcwd()}/../transformations")
import transformations

def stripleftchars(s):
    for i in range(len(s)):
        if s[i].isnumeric():
            return s[i:]
    return -1

def striplastchars(s):
    s = s.split("-")[-1]
    for i in range(len(s)):
        if s[i].isnumeric():
            return s[i:]
    return -1

class MultiDirectoryDataSequence(data.Dataset):
    def __init__(self, root=None, RRL_dir=None, image_size=(100,100), transform=None,
                 robustification=False, noise_level=10, sample_id="PREDICTION", max_dataset_size=None, effect=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.sample_id = sample_id
        self.root = root
        self.RRL_dir = RRL_dir
        self.effect = effect
        self.transform = transform
        self.robustification = robustification
        self.noise_level = noise_level
        self.size = 0
        self.image_size = image_size
        self.image_paths_hashmap = {}
        self.all_image_paths = []
        self.dfs_hashmap = {}
        self.dirs = []
        self.max_dataset_size = max_dataset_size
        if self.root is not None:
            self.process_basemodel_dirs()
        if RRL_dir is not None:
            self.process_RRL_dir()
        self.cache = {}
        self.printed = False

    def process_RRL_dir(self):
        skipcount = 0
        if self.RRL_dir is not None and Path(self.RRL_dir).is_dir():
            for p in Path(self.RRL_dir).iterdir():
                if p.is_dir() and "tb_logs_DDPG" not in str(p):
                    # ep = int(str(p.stem).replace("ep", ""))
                    # if ep % 8 != 0:
                    self.dirs.append("{}/{}".format(p.parent, p.stem))
                    image_paths = []
                    try:
                        self.dfs_hashmap[f"{p}"] = pd.read_csv(f"{p}/data.csv")
                    except FileNotFoundError as e:
                        try:
                            self.dfs_hashmap[f"{p}"] = pd.read_csv(f"{p}/data.txt")
                        except FileNotFoundError as e:
                            print(e, "\nNo data.csv or data.txt in directory")
                            continue
                    for pp in Path(p).iterdir():
                        if pp.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"] and "sample-base" in pp.name:
                            image_paths.append(pp)
                            self.all_image_paths.append(pp)
                    image_paths.sort(key=lambda p: int(striplastchars(p.stem)))
                    self.image_paths_hashmap[p] = copy.deepcopy(image_paths)
                    self.size += len(image_paths)
        print(f"Full dataset size: {self.size=}")
        print(f"{self.max_dataset_size=}")
        if self.max_dataset_size is not None:
            if self.size < self.max_dataset_size:
                print(f"Dataset size is {self.size} ({self.max_dataset_size - self.size} less than {self.max_dataset_size=}. Not limiting dataset size.)")
                exit(0)
            else:
                ratio = (self.size - self.max_dataset_size) / self.size 
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
                print(f"{dropped_dataset_size=}")
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
                image_base_rb, image_transf_rb, y_steer_rb = self.robustify(image_base, image_transf, y_steer)
                return {"image_base": image_base_rb, "image_transf": image_transf_rb, "steering_input": y_steer_rb, "throttle_input": sample["throttle_input"]} #,  "img_name": sample["img_name"], "all": torch.FloatTensor([y_steer, sample["throttle_input"]])}
            else:
                return self.cache[idx]
        img_name = self.all_image_paths[idx]
        image = Image.open(img_name)
        image_base = image.resize(self.image_size)
        image_base = self.transform(image_base)
        # Load "new hardware" image (true transform image) from dataset into image_transf
        if self.effect is not None:
            if self.effect == "fisheye":
                img_name_transf = str(img_name).replace("base", "transf")
                image_transf = Image.open(img_name_transf)
                image_transf = image_transf.resize(self.image_size)
            elif self.effect == "resdec":
                img_name_transf = str(img_name).replace("base", "lores")
                image_transf = Image.open(img_name_transf)
                # print(f"{image_transf.size=}")
                image_transf = image_transf.resize(self.image_size)
                # print(f"After resdec resize {image_transf.size=}")
                # exit(0)
            elif self.effect == "resinc":
                img_name_transf = str(img_name).replace("base", "hires")
                image_transf = Image.open(img_name_transf)
                # print(f"{image_transf.size=}")
                image_transf = image_transf.resize(self.image_size)
                # print(f"After resinc resize {image_transf.size=}")
                # exit(0)
            elif self.effect == "depth":
                image_orig = Image.open(img_name)
                img_name_transf = str(img_name).replace("base", "depth")
                image_depth = Image.open(img_name_transf)
                image_transf = transformations.blur_with_depth_image(np.array(image_orig), np.array(image_depth))
                image_transf = Image.fromarray((image_transf * 255).astype(np.uint8)).resize(self.image_size)
        else:
            img_name_transf = str(img_name).replace("base", "transf")
            image_transf = Image.open(img_name_transf)
            image_transf = image_transf.resize(self.image_size)
        image_transf = self.transform(image_transf)
        orig_image = torch.clone(image_base)
        pathobj = Path(img_name)
        df = self.dfs_hashmap[f"{pathobj.parent}"]
        df_index = df.index[df['IMG'] == img_name.name]
        # print(df.loc[df_index, self.sample_id])
        orig_y_steer = df.loc[df_index, self.sample_id].item()
        y_throttle = df.loc[df_index, 'THROTTLE_INPUT'].item()
        y_steer = copy.deepcopy(orig_y_steer)
        
        if self.robustification:
            image_base_rb, image_transf_rb, y_steer_rb = self.robustify(copy.deepcopy(image_base), copy.deepcopy(image_transf), y_steer)
            sample =  {"image_base": image_base_rb, "image_transf": image_transf_rb, "steering_input": y_steer_rb, "throttle_input": y_throttle }
        else:
            sample =  {"image_base": image_base, "image_transf": image_transf, "steering_input": orig_y_steer, "throttle_input": y_throttle}
        
        orig_sample =  {"image_base": image_base, "image_transf": image_transf, "steering_input": orig_y_steer, "throttle_input": y_throttle} #,  "img_name": img_name, "all": torch.FloatTensor([y_steer, y_throttle])}
        
        if sys.getsizeof(self.cache) < 8 * 1.0e10:
            self.cache[idx] = orig_sample
        elif not self.printed:
            print(f"{len(self.cache.keys())=}")
            self.printed = True
        return sample

    def get_outputs_distribution(self):
        all_outputs = np.array([])
        for key in self.dfs_hashmap.keys():
            df = self.dfs_hashmap[key]
            arr = df[self.sample_id].to_numpy()
            all_outputs = np.concatenate((all_outputs, arr), axis=0)
        all_outputs = np.array(all_outputs)
        moments = self.get_distribution_moments(all_outputs)
        return moments

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
