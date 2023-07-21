import numpy as np
import discorpy.losa.loadersaver as io
import discorpy.post.postprocessing as post
import os 
from PIL import Image
import cv2
import transforms
import detransformations


def deblur_IFAN(imgfile="/p/sdbb/supervised-transformation-dataset-alltransforms3/jungle_rock_island-8000-extra_jungle8000-fisheye.None-run00-6_13-23_45-7ZYMOA/sample-transf-00586.jpg"):
    output_base = "figs_deblur/"
    if not os.path.exists(output_base):
        os.makedirs(output_base, exist_ok=True)

    # mat0 = cv2.imread(imgfile) #, cv2.IMREAD_COLOR)
    mat0 = Image.open(imgfile)
    mat0 = mat0.convert('RGB') 
    mat0 = np.array(mat0)
    print(f"{np.max(mat0)=}")
    if np.max(mat0) > 1.0:
        mat0 = mat0 / 255
    mat0 = np.float32(mat0)

    out_orig = Image.fromarray((mat0 * 255).astype(np.uint8))
    out_orig.save(output_base + "orig_" + imgfile.split("/")[-1])
    # cv2.imwrite(output_base + "orig_" + imgfile.split("/")[-1], mat0)

    (height, width, channels) = mat0.shape

    img_deblurred = detransforms.deblur(mat0)
    print(f"{np.max(mat0)=}\n")
    print(f"{img_deblurred.shape=}")
    print(f"{type(img_deblurred)=}")
    
    img_deblurred = Image.fromarray(img_deblurred)
    img_deblurred = img_deblurred.convert('RGB') 
    img_deblurred.save(output_base + imgfile.split("/")[-1])
    # cv2.imwrite(output_base + imgfile.split("/")[-1], img_deblurred)
    return img_deblurred

def deblur_IFAN_orig(imgfile):
    img_deblurred = detransforms.deblur_test(imgfile)
    print(f"Saved to {img_deblurred}")

# deblur_IFAN_orig(imgfile="/p/sdbb/supervised-transformation-dataset-alltransforms3/automation_test_track-7736-straightwidehighway-fisheye.None-run00-6_14-9_37-7ZYMOA/sample-transf-00347.jpg")
# deblur_IFAN_orig(imgfile="/p/sdbb/discorpy/examples/Perseverance_distortion_correction/Sol0_1st_color.png")
# deblur_IFAN_orig(imgfile="/p/sdbb/supervised-transformation-dataset-alltransforms3/utah-14963-extra_utahlong2-fisheye.None-run00-6_13-14_11-7ZYMOA/sample-transf-01947.jpg")

# deblur_IFAN(imgfile="/p/sdbb/supervised-transformation-dataset-alltransforms3/automation_test_track-7736-straightwidehighway-fisheye.None-run00-6_14-9_37-7ZYMOA/sample-transf-00347.jpg")
# deblur_IFAN(imgfile="/p/sdbb/discorpy/examples/Perseverance_distortion_correction/Sol0_1st_color.png")
# deblur_IFAN(imgfile="/p/sdbb/supervised-transformation-dataset-alltransforms3/utah-14963-extra_utahlong2-fisheye.None-run00-6_13-14_11-7ZYMOA/sample-transf-01947.jpg")

deblur_IFAN(imgfile="/p/sdbb/testimgs-setdepths/sample-base-00010.jpg")
deblur_IFAN(imgfile="/p/sdbb/testimgs-setdepths/sample-base-00020.jpg")
deblur_IFAN(imgfile="/p/sdbb/testimgs-setdepths/sample-base-00030.jpg")
deblur_IFAN(imgfile="/p/sdbb/testimgs-setdepths/sample-base-00040.jpg")
deblur_IFAN(imgfile="/p/sdbb/testimgs-setdepths/sample-base-00050.jpg")
deblur_IFAN(imgfile="/p/sdbb/testimgs-setdepths/sample-base-00060.jpg")
deblur_IFAN(imgfile="/p/sdbb/testimgs-setdepths/sample-base-00070.jpg")
deblur_IFAN(imgfile="/p/sdbb/testimgs-setdepths/sample-base-00080.jpg")
deblur_IFAN(imgfile="/p/sdbb/testimgs-setdepths/sample-base-00090.jpg")
deblur_IFAN(imgfile="/p/sdbb/testimgs-setdepths/sample-base-00100.jpg")