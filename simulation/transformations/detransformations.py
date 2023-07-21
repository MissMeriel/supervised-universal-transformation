# assume img passed in as numpy file?
import numpy as np
import torch
import numpy as np
import discorpy.losa.loadersaver as io
import discorpy.post.postprocessing as post
import os 
from PIL import Image
import cv2
import skimage
import sys

sys.path.append("/p/sdbb/vqvae/IFAN")
from predict import Predictor
predictor = Predictor()
predictor.setup()

def deblur(img):
    # convert to tensor
    # call IFAN model
    img_deblurred = predictor.predict_image(img)
    return img_deblurred

def deblur_test(imgfile):
    # convert to tensor
    # call IFAN model
    img_deblurred = predictor.predict(imgfile)
    return img_deblurred

def defisheye(img):
    list_pow = np.asarray([10**(0), 10**(-3), 10**(-6), 10**(-7), 10**(-12)])
    list_coef = np.asarray([1.0, 7.0, 10.5, 7.0, 3.0])
    (height, width, channels) = img.shape
    img = img / np.max(img)

    # Estimated forward model
    xcenter = width / 2.0
    ycenter = height / 2.0
    list_ffact = list_pow * list_coef

    # Calculate parameters of a backward model from the estimated forward model
    list_hor_lines = []
    for i in range(20, height-20, 50):
        list_tmp = []
        for j in range(20, width-20, 50):
            list_tmp.append([i - ycenter, j - xcenter])
        list_hor_lines.append(list_tmp)
    Amatrix = []
    Bmatrix = []
    list_expo = np.arange(len(list_ffact), dtype=np.int16)
    for _, line in enumerate(list_hor_lines):
        for _, point in enumerate(line):
            xd = np.float64(point[1])
            yd = np.float64(point[0])
            rd = np.sqrt(xd * xd + yd * yd)
            ffactor = np.float64(np.sum(list_ffact * np.power(rd, list_expo)))
            if ffactor != 0.0:
                Fb = 1 / ffactor
                ru = ffactor * rd
                Amatrix.append(np.power(ru, list_expo))
                Bmatrix.append(Fb)
    Amatrix = np.asarray(Amatrix, dtype=np.float64)
    Bmatrix = np.asarray(Bmatrix, dtype=np.float64)
    list_bfact = np.linalg.lstsq(Amatrix, Bmatrix, rcond=1e-64)[0]

    # Apply distortion correction
    corrected_mat = np.zeros(img.shape)
    for i in range(img.shape[-1]):
        corrected_mat[:, :, i] = post.unwarp_image_backward(img[:, :, i], xcenter, ycenter, list_bfact)
    return corrected_mat
    
def resize_cv2(img, shape=(135,240)):
    res = cv2.resize(img, dsize=shape, interpolation=cv2.INTER_CUBIC)
    return res

def resize_skimage(img, shape=(135,240)):
    img = skimage.transform.resize(img, shape)
    return img

def resize_PIL(img, shape=(135,240)):
    img_pil = Image.fromarray(img)
    img_pil = img_pil.resize(shape)
    return np.array(img_pil)


# def fisheye_inv(image):
#     with WandImage.from_array(image) as img:
#         img.virtual_pixel = 'transparent'
#         img.distort('barrel_inverse', (0.0, 0.0, -0.5, 1.5))
#         img = np.array(img, dtype='uint8')
#         return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)