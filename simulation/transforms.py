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
        sigma = (10 * depth) / 256
        # print(f"{depth=} \t{sigma=}")
        img_blurred = gaussian_filter(img_base, sigma=(sigma, sigma, 0)) # fix for dtype=float
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
from predict import Predictor

predictor = Predictor()
predictor.setup()

def transform_deblur(img):
    img_deblurred = predictor.predict(img)
    return img_deblurred



def generate_blurry_imgs():
    dir = "F:/supervised-transformation-dataset-alltransforms/automation_test_track-8290-Rturnrockylinedmtnroad-fisheye.None-run00-6_4-23_39-7ZYMOA"
    dir = "F:/supervised-transformation-dataset-alltransforms3/utah-14912-extra_utahexittunnelright-fisheye.None-run00-6_13-12_25-7ZYMOA"
    dir= "F:/supervised-transformation-dataset-alltransforms3/west_coast_usa-12641-extra_dock-fisheye.None-run00-6_13-12_36-7ZYMOA"
    outdir = "./testimgs/"

    os.makedirs(outdir, exist_ok=True)
    for i, file in enumerate(os.listdir(dir)):
        # if "data" not in file:
        if "sample-base" in file and i % 10 == 0 and i <= 100:
            print(f"{file=}")
            img_base = Image.open(dir+"/"+file)
            img_depth = Image.open(dir+"/"+file.replace("base", "depth"))
            img_adjusteddof = transform_depthoffield(img_base, img_depth)
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
    deblur()