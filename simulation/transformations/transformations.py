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

def fisheye(img:np.ndarray) -> np.ndarray:
    # transpose channels?
    # barrel transformation from discorpy
    pass
    

# def resize(img:np.ndarray, shape=(135,240)) -> np.ndarray:
def resize(img: np.ndarray, shape=(192, 108)) -> np.ndarray:
    img_pil = Image.fromarray(img).resize(shape)
    return np.array(img_pil)


# for resolution increase on low quality/small images
def resize_with_inpainting(img:np.ndarray, shape=(135,240)) -> np.ndarray:
    pass

# for resolution increase on low quality/small images
def resize_with_superresolution(img:np.ndarray, shape=(135,240)) -> np.ndarray:
    pass


# def fisheye_wand(image, filename=None):
#     with WandImage.from_array(image) as img:
#         img.virtual_pixel = 'transparent'
#         img.distort('barrel', (0.1, 0.0, -0.05, 1.0))
#         img.alpha_channel = False
#         img = np.array(img, dtype='uint8')
#         return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
