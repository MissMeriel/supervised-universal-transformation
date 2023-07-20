import numpy as np
import discorpy.losa.loadersaver as io
import discorpy.post.postprocessing as post
import os 
from PIL import Image
from matplotlib import pyplot as plt

list_pow = np.asarray([1.0, 10**(-4), 10**(-7), 10**(-10), 10**(-13)])
list_pow = np.asarray([10**(0), 10**(-3), 10**(-6), 10**(-7), 10**(-12)])
# list_pow = np.asarray([10**(0), 10**(-1), 10**(-2), 10**(-3), 10**(-4)]) # kind of works, too small & flipped
list_pow = np.asarray([10**(0), 10**(-3), 10**(-5), 10**(-11), 10**(-13)])
list_coef = np.asarray([1.0, 7.0, 8.0, 4.0, 3.0])

# def inpaint_black(img:np.ndarray):



def estimate_distortion(imgfile="/p/sdbb/supervised-transformation-dataset-alltransforms3/jungle_rock_island-8000-extra_jungle8000-fisheye.None-run00-6_13-23_45-7ZYMOA/sample-transf-00586.jpg", output_base="figs/"):
    # load image as 1C np.ndarray
    mat0 = io.load_image(imgfile, average=False)

    if not os.path.exists(output_base):
        os.makedirs(output_base, exist_ok=True)

    (height, width, channels) = mat0.shape
    if np.max(mat0) > 1.0:
        mat0 = mat0 / np.max(mat0)

    # Create a line-pattern image
    line_pattern = np.zeros((height, width, channels), dtype=np.float32)
    for i in range(50, height - 10, 5):
        line_pattern[i - 1:i + 1] = 1.0

    # Estimate parameters by visual inspection.
    xcenter = width / 2.0
    ycenter = height / 2.0
    list_ffact = list_pow * list_coef

    pad = int(width / 100)
    mat_pad = np.pad(mat0, ((pad, pad), (pad, pad), (0,0)), mode='constant')
    print(f"{line_pattern.shape=}")
    print(f"{mat0.shape=}")
    mat0 = mat_pad
    # print(f"{mat_pad.shape=}")
    # io.save_image(output_base + "/mat_pad.jpg", (mat_pad * 255).astype(np.uint8))
    # mat_cor = post.unwarp_image_backward(mat_pad, xcenter + pad, ycenter + pad, list_ffact)
    mat_cor = np.zeros(mat0.shape)
    for i in range(mat0.shape[-1]):
        mat_cor[:, :, i] = post.unwarp_image_backward(mat0[:, :, i], xcenter, ycenter, list_ffact)

    # mat_cor = mat_cor[pad:pad + height, pad:pad + width]
    print(f"{mat_cor.shape=}")
    io.save_image(output_base + "/overlay.jpg", (mat0 + 0.75*mat_cor))
    io.save_image(output_base + "/mat_cor.jpg", mat_cor)
    sim_img = Image.open(imgfile.replace("base", "transf"))
    orig_img = Image.open(imgfile)
    fig = plt.figure()
    ax1 = fig.add_subplot(1,3,1)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.imshow(orig_img)
    ax1.set_title("CAM0 image")
    ax2 = fig.add_subplot(1,3,2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.imshow(mat_cor)
    ax2.set_title("Transformed image")
    ax3 = fig.add_subplot(1,3,3)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.imshow(np.array(sim_img))
    ax3.set_title("CAM1 image")
    fig.savefig(output_base + "/side_by_side1.png")
    # plt.savefig(output_base + "side_by_side.png")

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

    # pad and inpaint
    print(f"{np.max(corrected_mat)=}")
    img_pil = Image.fromarray(np.array(corrected_mat * 255, dtype=np.uint8)).resize(( 192,108))
    corrected_mat = np.array(img_pil)
    padx = int((mat0.shape[0] - corrected_mat.shape[0]) / 2)
    pady = int((mat0.shape[1] - corrected_mat.shape[1]) / 2)
    corrected_mat = np.pad(corrected_mat, ((padx, padx), (pady, pady), (0, 0)), mode='edge')
    print(f"{corrected_mat.shape=}")
    # plot side by side of transformation and actual img from sim
    sim_img = Image.open(imgfile.replace("base", "transf"))
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(corrected_mat)
    ax1.set_title("Transformed image")
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(np.array(sim_img))
    ax2.set_title("Sim. image")
    fig.savefig(output_base + "/side_by_side.png")
    # plt.savefig(output_base + "side_by_side.png")

# estimate_distortion(imgfile="/p/sdbb/supervised-transformation-dataset-alltransforms3/automation_test_track-7736-straightwidehighway-fisheye.None-run00-6_14-9_37-7ZYMOA/sample-transf-00347.jpg", output_base="./figs")
# estimate_distortion(imgfile="/p/sdbb/supervised-transformation-dataset-alltransforms3/utah-14963-extra_utahlong2-fisheye.None-run00-6_13-14_11-7ZYMOA/sample-transf-01947.jpg", output_base="./figs")
estimate_distortion(imgfile="/p/sdbb/supervised-transformation-dataset-alltransforms3/utah-14963-extra_utahlong2-fisheye.None-run00-6_13-14_11-7ZYMOA/sample-base-01947.jpg", output_base="./figs_fish")
estimate_coefficients(imgfile="/p/sdbb/supervised-transformation-dataset-alltransforms3/utah-14963-extra_utahlong2-fisheye.None-run00-6_13-14_11-7ZYMOA/sample-base-01947.jpg", output_base="./figs_fish")
