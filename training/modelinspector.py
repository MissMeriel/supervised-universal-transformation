import os
import os.path
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
import logging
import scipy.misc
import copy
import torch
import statistics, math
import csv
from ast import literal_eval
import PIL
import sys
sys.path.append(f'{os.getcwd()}/../models')
from DAVE2pytorch import DAVE2PytorchModel, DAVE2v3
from ResNet import ResNet50, ResNet101, ResNet152
from torchvision.transforms import Compose, ToPILImage, ToTensor

def get_transf(transf_id):
    if transf_id is None:
        img_dims = (240,135); fov = 51; transf = "None"
    elif "fisheye" in transf_id:
        img_dims = (240,135); fov=75; transf = "fisheye"
    elif "resdec" in transf_id:
        img_dims = (96, 54); fov = 51; transf = "resdec"
    elif "resinc" in transf_id:
        img_dims = (480,270); fov = 51; transf = "resinc"
    elif "depth" in transf_id:
        img_dims = (240, 135); fov = 51; transf = "depth"
    return img_dims, fov, transf

def is_img(i):
    return ("jpg" in i and "annot" not in i)

def main():
    global base_filename
    model_name = "../weights/model-DAVE2v3-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt" # orig model
    # model_name = "../models/retrained-lighterblur-noflip-fixednoise/model-fixnoise-DAVE2v3-135x240-lr1e4-100epoch-64batch-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-noiseflipblur-best.pt"
    # model_name = "../models/weights/model-ResNet-randomblurnoise-135x240-lr1e4-500epoch-64batch-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-noiseflipblur-epoch121.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = torch.load(model_name, map_location=device).eval()
    print(model)
    # test_img_dir = "F:/old-RRL-results/CAMtest-DDPG-normimg-0.05eps-1_16-11_31-WXKX04"
    test_img_dir = "F:/old-RRL-results/CAMtest-DDPG-fov120-1_14-13_35-0C8TWP"
    # img = cv2.imread(f"{test_img_dir}/imgmain-epi001-step0000.jpg")
    from PIL import Image
    imgs = [i for i in os.listdir(test_img_dir) if "imgmain" in i]
    img_transf = [i for i in os.listdir(test_img_dir) if "imgtest" in i]
    nonzero_counts_main, nonzero_counts_test = [], []
    for imgname1, imgname2 in zip(imgs, img_transf):
        img1 = Image.open(os.path.join(test_img_dir, imgname1)).resize((135, 240))
        img2 = Image.open(os.path.join(test_img_dir, imgname2)).resize((135, 240))
        img1 = model.process_image(img1)
        img2 = model.process_image(img2)
        theta1 = model(img1).item()
        features1 = model.features(img1)
        theta2 = model(img2).item()
        features2 = model.features(img2)
        nonzero_counts_main.append(torch.sum(features1 != 0))
        nonzero_counts_test.append(torch.sum(features2 != 0))
    # print(nonzero_counts_main)
    print(sum(nonzero_counts_main) / len(nonzero_counts_main))
    print(sum(nonzero_counts_test) / len(nonzero_counts_test))

if __name__ == '__main__':
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('PIL').setLevel(logging.WARNING)
    main()