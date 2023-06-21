from PIL import Image
import os, sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from tabulate import tabulate
from operator import itemgetter
dir = "F:/supervised-transformation-dataset-alltransforms3/"
table = []
total = 0
for subdir in os.listdir(dir):
    if os.path.isdir(dir + subdir):
        subcount = 0
        for file in os.listdir(dir + subdir):
            if "sample-base-" in file:
                subcount += 1
                total += 1
        table.append((subdir, subcount))

table = sorted(table, key=itemgetter(1))
print(tabulate(table, showindex="always"))
print("total", total)