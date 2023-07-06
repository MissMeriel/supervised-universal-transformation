from PIL import Image
import os, sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.filters import gaussian
import scipy.misc
import matplotlib.image

# generate shallow depth of field RGB image using depth image
def blur(img_base, img_depth):
    img_depth = np.array(img_depth, dtype=np.uint8)
    img_base = np.array(img_base) / 255
    # depths_img = np.unique(img_depth) # returns sorted
    depth_cutoffs = np.array([0, 3, 5, 256])
    # depth_cutoffs = np.array([0, 1, 2, 3, 4, 5, 256])
    img_new = np.zeros(img_base.shape)
    print(depth_cutoffs[:-1])
    # create len(depths) masks per image
    for i, depth in enumerate(depth_cutoffs[:-1]):
        # adjust blur according to the pixel value of the depth image
        sigma = np.sqrt(depth)
        print(f"{depth=} \t{i=} \t{depth_cutoffs[i+1]=} \t{depth_cutoffs=} \t{sigma=}")
        img_blurred = gaussian_filter(img_base, sigma=(sigma, sigma, 0)) # fix for dtype=float
        # img_blurred = gaussian(img_base, sigma=(sigma, sigma), channel_axis=3)
        # mask blurred image into aggregate image
        mask1 = img_depth >= depth
        mask2 = img_depth < depth_cutoffs[i+1]
        mask = np.multiply(mask1, mask2)
        img_mask = mask * img_blurred
        img_new += img_mask
        # plt.imshow(img_new)
        # plt.show()
        # plt.pause(0.01)
    img_new = np.clip(img_new, 0, 1)
    return img_new

# sys.path.append("../../IFAN")
# from predict import Predictor
#
# predictor = Predictor()
# predictor.setup()

def transform_deblur(img):
    img_deblurred = predictor.predict(img)
    return img_deblurred



def generate_blurry_imgs():
    dir = "F:/supervised-transformation-dataset-alltransforms/automation_test_track-8290-Rturnrockylinedmtnroad-fisheye.None-run00-6_4-23_39-7ZYMOA"
    dir = "F:/supervised-transformation-dataset-alltransforms3/utah-14912-extra_utahexittunnelright-fisheye.None-run00-6_13-12_25-7ZYMOA"
    dir= "F:/supervised-transformation-dataset-alltransforms3/west_coast_usa-12641-extra_dock-fisheye.None-run00-6_13-12_36-7ZYMOA"
    outdir = "./testimgs-setdepths/"

    os.makedirs(outdir, exist_ok=True)
    for i, file in enumerate(os.listdir(dir)):
        # if "data" not in file:
        if "sample-base" in file and i % 10 == 0 and i <= 100:
            print(f"{file=}")
            img_base = Image.open(dir+"/"+file)
            img_depth = Image.open(dir+"/"+file.replace("base", "depth"))
            img_adjusteddof = blur(img_base, img_depth)
            # Image.fromarray(img_adjusteddof).save(outdir+"/"+file)
            matplotlib.image.imsave(outdir+"/"+file, img_adjusteddof)

def deblur():
    indir = "./testimgs/"
    # indir= "F:/supervised-transformation-dataset-alltransforms3/west_coast_usa-12641-extra_dock-fisheye.None-run00-6_13-12_36-7ZYMOA"
    outdir = "./testimgsdeblurred/"
    os.makedirs(outdir, exist_ok=True)
    for i, file in enumerate(os.listdir(indir)):
        if "sample-base" in file and i % 10 == 0 and i <= 100:
            print(f"{file=}")
            # img_base = Image.open(indir + "/" + file)
            # img_deblurred = transform_deblur(img_base)
            img_deblurred = predictor.predict_img(indir + "/" + file)
            # matplotlib.image.imsave(outdir + "/" + file, np.array(img_deblurred))
            cv2.imwrite(str(outdir + "/" + file), img_deblurred)

if __name__ == '__main__':
    generate_blurry_imgs()
    # deblur()