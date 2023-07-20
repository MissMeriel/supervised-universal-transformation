import numpy as np
import discorpy.losa.loadersaver as io
import discorpy.post.postprocessing as post
import os 
from PIL import Image
import cv2
import transforms
from blurgenerator import motion_blur, lens_blur, gaussian_blur


def blur_computational(imgfile):
    output_base = "figs_blur_computational/"
    if not os.path.exists(output_base):
        os.makedirs(output_base, exist_ok=True)

    # mat0 = cv2.imread(imgfile) #, cv2.IMREAD_COLOR)
    mat0 = Image.open(imgfile)
    mat0 = mat0.convert('RGB')
    mat0 = np.array(mat0)
    # print(f"{np.max(mat0)=}")

    result = transforms.blur_computational(mat0)
    result = Image.fromarray(result)
    print(f"Saving result to {output_base + imgfile.split('/')[-1]}")
    result.save(output_base + imgfile.split('/')[-1])

# blur_learned_orig(imgfile="/p/sdbb/supervised-transformation-dataset-alltransforms3/automation_test_track-7736-straightwidehighway-fisheye.None-run00-6_14-9_37-7ZYMOA/sample-transf-00347.jpg")
# blur_learned_orig(imgfile="/p/sdbb/discorpy/examples/Perseverance_distortion_correction/Sol0_1st_color.png")
# blur_learned_orig(imgfile="/p/sdbb/supervised-transformation-dataset-alltransforms3/utah-14963-extra_utahlong2-fisheye.None-run00-6_13-14_11-7ZYMOA/sample-transf-01947.jpg")

blur_computational(imgfile="/p/sdbb/supervised-transformation-dataset-alltransforms3/automation_test_track-7736-straightwidehighway-fisheye.None-run00-6_14-9_37-7ZYMOA/sample-base-00347.jpg")
blur_computational(imgfile="/p/sdbb/discorpy/examples/Perseverance_distortion_correction/Sol0_1st_color.png")
blur_computational(imgfile="/p/sdbb/supervised-transformation-dataset-alltransforms3/utah-14963-extra_utahlong2-fisheye.None-run00-6_13-14_11-7ZYMOA/sample-transf-01947.jpg")

# blur_learned(imgfile="/p/sdbb/supervised-transformation-dataset-alltransforms3/automation_test_track-7736-straightwidehighway-fisheye.None-run00-6_14-9_37-7ZYMOA/sample-base-00010.jpg")
# blur_learned(imgfile="/p/sdbb/supervised-transformation-dataset-alltransforms3/automation_test_track-7736-straightwidehighway-fisheye.None-run00-6_14-9_37-7ZYMOA/sample-base-00050.jpg")
# blur_learned(imgfile="/p/sdbb/supervised-transformation-dataset-alltransforms3/automation_test_track-7736-straightwidehighway-fisheye.None-run00-6_14-9_37-7ZYMOA/sample-base-00100.jpg")
# blur_learned(imgfile="/p/sdbb/supervised-transformation-dataset-alltransforms3/automation_test_track-7736-straightwidehighway-fisheye.None-run00-6_14-9_37-7ZYMOA/sample-base-00150.jpg")
# blur_learned(imgfile="/p/sdbb/supervised-transformation-dataset-alltransforms3/automation_test_track-7736-straightwidehighway-fisheye.None-run00-6_14-9_37-7ZYMOA/sample-base-00200.jpg")
# blur_learned(imgfile="/p/sdbb/supervised-transformation-dataset-alltransforms3/automation_test_track-7736-straightwidehighway-fisheye.None-run00-6_14-9_37-7ZYMOA/sample-base-00250.jpg")