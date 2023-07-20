import numpy as np
import discorpy.losa.loadersaver as io
import discorpy.post.postprocessing as post
import os 
from PIL import Image
# combo 1
# list_pow =  np.asarray([1.0, 10**(-4), 10**(-7), 10**(-15), 10**(-20)])
# list_coef = np.asarray([1.0, 5.0, 9.0, 5.0, 1.0])

# combo 2
# list_pow = np.asarray([10**(0), 10**(-6), 10**(-7), 10**(-8), 10**(-9)])
# list_coef = np.asarray([1.0, 3.0, 7.0, 3.0, 1.0])

# combo 3 (best)
# list_pow = np.asarray([10**(0), 10**(-6), 10**(-7), 10**(-8), 10**(-9)])
# list_coef = np.asarray([1.0, 4.0, 5.0, 17.0, 3.0])
# list_coef = np.asarray([1.0, 10.0, 36.0, 17.0, 3.0])

# combo 4 (car finally out of pic)
# list_pow = np.asarray([10**(0), 10**(-3), 10**(-6), 10**(-7), 10**(-12)])
# list_coef = np.asarray([1.0, 20.0, 50.0, 17.0, 3.0])

# combo 5 (car hood almost back in pic)
# list_pow = np.asarray([10**(0), 10**(-3), 10**(-6), 10**(-7), 10**(-12)])
# list_coef = np.asarray([1.0, 11.0, 12.0, 10.0, 3.0])
# or
# list_pow = np.asarray([10**(0), 10**(-3), 10**(-6), 10**(-7), 10**(-12)])
# list_coef = np.asarray([1.0, 7.0, 10.5, 7.0, 3.0])

list_pow = np.asarray([1.0, 10**(-4), 10**(-7), 10**(-10), 10**(-13)])
list_pow = np.asarray([10**(0), 10**(-3), 10**(-6), 10**(-7), 10**(-12)])
list_coef = np.asarray([1.0, 7.0, 10.5, 7.0, 3.0])
def estimate_distortion(imgfile="/p/sdbb/supervised-transformation-dataset-alltransforms3/jungle_rock_island-8000-extra_jungle8000-fisheye.None-run00-6_13-23_45-7ZYMOA/sample-transf-00586.jpg"):
    # load image as 1C np.ndarray
    mat0 = io.load_image(imgfile, average=False)
    output_base = "figs/"

    if not os.path.exists(output_base):
        os.makedirs(output_base, exist_ok=True)

    (height, width, channels) = mat0.shape
    mat0 = mat0 / np.max(mat0)

    # Create a line-pattern image
    line_pattern = np.zeros((height, width, channels), dtype=np.float32)
    for i in range(50, height - 10, 5):
        line_pattern[i - 1:i + 1] = 1.0

    # Estimate parameters by visual inspection.
    # Coarse estimation
    xcenter = width / 2.0
    ycenter = height / 2.0
    # list_pow = np.asarray([1.0, 10**(-4), 10**(-7), 10**(-10), 10**(-13)])
    # list_pow = np.asarray([1.0, 10**(-5), 10**(-10), 10**(-25), 10**(-50)])
    # Fine estimation
    # list_coef = np.asarray([3.0, 50.0, 100.0, 50.0, 3.0])
    list_ffact = list_pow * list_coef
    print(f"{list_ffact=}")

    pad = width
    mat_pad = np.pad(line_pattern, pad, mode='edge')
    print(f"{line_pattern.shape=}")
    print(f"{mat0.shape=}")
    print(f"{mat_pad.shape=}")
    # io.save_image(output_base + "/mat_pad.jpg", (mat_pad * 255).astype(np.uint8))
    # mat_cor = post.unwarp_image_backward(mat_pad, xcenter + pad, ycenter + pad, list_ffact)

    mat_cor = mat_cor[pad:pad + height, pad:pad + width]
    print(f"{mat_cor.shape=}")
    io.save_image(output_base + "/overlay.jpg", (mat0 + 0.5*mat_cor))
    io.save_image(output_base + "/mat_cor.jpg", mat_cor)

def estimate_coefficients(imgfile="/p/sdbb/discorpy/examples/Perseverance_distortion_correction/Sol0_1st_color.png", 
                          output_base="/p/sdbb/discorpy/examples/Perseverance_distortion_correction/figs/"):    
    # Load image
    mat0 = io.load_image(imgfile, average=False)
    (height, width, channels) = mat0.shape
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
    print(f"{mat0.shape=}")
    corrected_mat = np.zeros(mat0.shape)
    for i in range(mat0.shape[-1]):
        corrected_mat[:, :, i] = post.unwarp_image_backward(mat0[:, :, i], xcenter, ycenter, list_bfact)
    io.save_image(output_base + "/after.png", corrected_mat)
    io.save_image(output_base + "/before.png", mat0)
    io.save_metadata_txt(output_base + "/coefficients.txt", xcenter, ycenter, list_bfact)

# estimate_distortion(imgfile="/p/sdbb/supervised-transformation-dataset-alltransforms3/automation_test_track-7736-straightwidehighway-fisheye.None-run00-6_14-9_37-7ZYMOA/sample-transf-00347.jpg")
# estimate_distortion(imgfile="/p/sdbb/discorpy/examples/Perseverance_distortion_correction/Sol0_1st_color.png")
# estimate_distortion(imgfile="/p/sdbb/supervised-transformation-dataset-alltransforms3/utah-14963-extra_utahlong2-fisheye.None-run00-6_13-14_11-7ZYMOA/sample-transf-01947.jpg")
# estimate_coefficients()
# estimate_coefficients(imgfile="/p/sdbb/supervised-transformation-dataset-alltransforms3/utah-14963-extra_utahlong2-fisheye.None-run00-6_13-14_11-7ZYMOA/sample-transf-01947.jpg", 
#                       output_base="./figs")
