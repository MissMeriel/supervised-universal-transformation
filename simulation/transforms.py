from PIL import Image
import os, sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import scipy.misc
import matplotlib.image

def transform_depthoffield(img_base, img_depth):
    img_depth = np.array(img_depth, dtype=np.uint8)
    img_base = np.array(img_base) / 256
    depths = np.unique(img_depth) # returns sorted
    print(f"{depths.shape=}")
    img_new = np.zeros(img_base.shape, dtype=float)
    # create len(depths) masks per image
    for depth in depths:
        # adjust blur according to the pixel value of the depth image
        sigma = (7 * depth) / 256
        print(f"{depth=} \t{sigma=}")
        img_blurred = gaussian_filter(img_base, sigma=(sigma, sigma, 1)) # fix for dtype=float
        # plt.imshow(img_blurred)
        # plt.show()
        # plt.pause(0.1)
        # mask blurred image into aggregate image
        mask = img_depth == depth
        img_mask = mask * img_blurred
        img_new += img_mask
        # plt.imshow(img_new)
        # plt.show()
        # plt.pause(0.01)
    # account for the edges of masks ?
    return img_new

sys.path.append("../../IFAN")
from IFAN.predict import predict
def transform_deblur(img):

    return img_deblurred

dir = "F:/supervised-transformation-dataset-alltransforms/automation_test_track-8290-Rturnrockylinedmtnroad-fisheye.None-run00-6_4-23_39-7ZYMOA"
dir = "F:/supervised-transformation-dataset-alltransforms/automation_test_track-8396-test1-fisheye.None-run00-6_4-21_54-7ZYMOA"
outdir = "./testimgs/"
os.makedirs(outdir, exist_ok=True)
for file in os.listdir(dir):
    print(f"{file=}")
    if "data" not in file:
        if "sample-base" in file:
            img_base = Image.open(dir+"/"+file)
            img_depth = Image.open(dir+"/"+file.replace("base", "depth"))
            img_adjusteddof = transform_depthoffield(img_base, img_depth)
            # Image.fromarray(img_adjusteddof).save(outdir+"/"+file)
            matplotlib.image.imsave(outdir+"/"+file, img_adjusteddof)