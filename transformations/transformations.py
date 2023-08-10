# assume img passed in as numpy file
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from blurgenerator import motion_blur, lens_blur, gaussian_blur


# generate shallow depth of field RGB image using depth image
def blur_with_depth_image(img_base:np.ndarray, img_depth:np.ndarray) -> np.ndarray:
    img_depth = np.array(img_depth, dtype=np.uint8)
    if np.max(img_base) > 1.:
        img_base = np.array(img_base) / 255
    depth_cutoffs = np.array([0, 3, 5, 256])
    img_new = np.zeros(img_base.shape)
    # create len(depths) masks per image
    for i, depth in enumerate(depth_cutoffs[:-1]):
        # adjust blur according to the pixel value of the depth image
        sigma = np.sqrt(depth)
        img_blurred = gaussian_filter(img_base, sigma=(sigma, sigma, 0)) # fix for dtype=float
        # mask blurred image into aggregate image
        mask1 = img_depth >= depth
        mask2 = img_depth < depth_cutoffs[i+1]
        mask = np.multiply(mask1, mask2)
        img_mask = mask * img_blurred
        img_new += img_mask
    img_new = np.clip(img_new, 0, 1)
    return img_new

def blur_computational(img:np.ndarray) -> np.ndarray:
    # result = gaussian_blur(mat0, 10)
    result = lens_blur(img, 4)
    return result

def blur_bokeh(img:np.ndarray) -> np.ndarray:
    pass


import discorpy.losa.loadersaver as io
import discorpy.post.postprocessing as post

def fisheye(mat0:np.ndarray) -> np.ndarray:
    list_pow = np.asarray([10**(0), 10**(-3), 10**(-5), 10**(-11), 10**(-13)])
    list_coef = np.asarray([1.0, 7.0, 8.0, 4.0, 3.0])
    (height, width, channels) = mat0.shape
    if np.max(mat0) > 1.0:
        mat0 = mat0 / np.max(mat0)

    # Estimated forward model
    xcenter = width / 2.0
    ycenter = height / 2.0
    # list_pow = np.asarray([1.0, 10**(-4), 10**(-7), 10**(-10), 10**(-13)])
    # list_coef = np.asarray([1.0, 4.0, 5.0, 17.0, 3.0])
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
    # corrected_mat = post.unwarp_image_backward(mat0, xcenter, ycenter, list_bfact)
    # print(f"{mat0.shape=}")
    corrected_mat = np.zeros(mat0.shape)
    for i in range(mat0.shape[-1]):
        corrected_mat[:, :, i] = post.unwarp_image_backward(mat0[:, :, i], xcenter, ycenter, list_bfact)
    return corrected_mat
    

def resize(img:np.ndarray, shape=(135,240)) -> np.ndarray:
    img_pil = Image.fromarray(img).resize(shape)
    return np.array(img_pil)

def resize_pil(img:Image, shape=(135,240)) -> np.ndarray:
    return img.resize(shape)

# for resolution increase on low quality/small images
def resize_with_inpainting(img:np.ndarray, shape=(135,240)) -> np.ndarray:
    # convert to PIL
    # resize using PIL Image.resize()
    pass