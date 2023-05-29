import numpy as np
import cv2
from PIL import Image
from .matlab_cp2tform import get_similarity_transform_for_cv2

DEFAULT_CROP_SIZE = (96, 112)

def reference_landmark():
    return np.array([[38.29459953, 51.69630051],
                     [73.53179932, 51.50139999],
                     [56.02519989, 71.73660278],
                     [41.54930115, 92.3655014],
                     [70.72990036, 92.20410156]])


def warp_and_crop_face(src_img,
                       facial_pts,
                       reference_pts=None,
                       crop_size=(112, 112),
                       ):

    ref_pts = np.float32(reference_pts)
    ref_pts_shp = ref_pts.shape

    if ref_pts_shp[0] == 2:
        ref_pts = ref_pts.T

    src_pts = np.float32(facial_pts)
    src_pts_shp = src_pts.shape

    if src_pts_shp[0] == 2:
        src_pts = src_pts.T

    tfm = get_similarity_transform_for_cv2(src_pts, ref_pts)
    face_img = cv2.warpAffine(np.array(src_img), tfm, (crop_size[0], crop_size[1]))
    face_img = Image.fromarray(face_img)

    return face_img, tfm