import torch.nn.functional as F
import torchvision
import skimage.transform as trans
from PIL import Image
from torchvision.transforms import ToTensor
import cv2
import torch
import numpy as np


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = (inp * 255).astype(np.uint8)
    return inp


def transform_keypoints(kps, meta, invert=False):
    keypoints = kps.copy()
    if invert:
        meta = np.linalg.inv(meta)
    keypoints[:, :2] = np.dot(keypoints[:, :2], meta[:2, :2].T) + meta[:2, 2]
    return keypoints


def cv2_param_to_torch_theta(cv2_tfm, image_width, image_height, output_width, output_height):
    # https://github.com/wuneng/WarpAffine2GridSample
    """4.Affine Transformation Matrix to theta"""
    src = np.array([[0, 0], [0, 1], [1, 1]], dtype=np.float32)
    dst = transform_keypoints(src, cv2_tfm)

    # normalize to [-1, 1]
    src = src / [image_width, image_height] * 2 - 1
    dst = dst / [output_width, output_height] * 2 - 1
    theta = trans.estimate_transform("affine", src=dst, dst=src).params
    theta = theta[:2].astype(np.float32)
    return theta


if __name__ == '__main__':

    raw = '/data/data/faces/casia_webface/raw/CASIA-WebFace/2207150/005.jpg'
    aligned = '/data/data/faces/casia_webface/raw/CASIA-WebFace_raw_aligned_mtcnn/2207150/005.jpg'
    image = np.array(Image.open(raw))
    image_gt = np.array(Image.open(aligned))

    # null transform
    cv2_tfm = np.array([[1, 0, 0],
                        [0, 1, 0]]).astype(np.float32)
    cv_img = cv2.warpAffine(image_gt, cv2_tfm[:2], (112,112))
    cv2.imwrite('/mckim/temp/null.png', cv_img)
    cv2_param_to_torch_theta(cv2_tfm, 112, 112, 112, 112)


    image_width = 250
    image_height = 250
    output_width = 112
    output_height = 112
    output_size = (output_width, output_height)

    cv2_tfm = np.array([[0.82096402, -0.07303078, -23.32888536],
                        [0.07303078, 0.82096402, -48.49860474]]).astype(np.float32)
    cv_img = cv2.warpAffine(image, cv2_tfm[:2], output_size)
    cv2.imwrite('/mckim/temp/cv2.png', cv_img)

    theta = cv2_param_to_torch_theta(cv2_tfm, image_width, image_height, output_width, output_height)

    to_tensor = torchvision.transforms.ToTensor()
    tensor = to_tensor(image).unsqueeze(0)
    theta = torch.tensor(theta, dtype=torch.float32).unsqueeze(0)
    output_size = torch.Size((1, 3, output_height, output_width))
    grid = F.affine_grid(theta[:, :2], output_size, align_corners=True)
    tensor = F.grid_sample(tensor, grid, align_corners=True)
    tensor = tensor.squeeze(0)
    torch_img = convert_image_np(tensor)
    print("Torch image size: (%d, %d)" % (torch_img.shape[1], torch_img.shape[0]))
    cv2.imwrite('/mckim/temp/torch.png', torch_img)

    cv2.imwrite('/mckim/temp/image_gt.png', image_gt)