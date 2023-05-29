import sys

from PIL import Image
import numpy as np
import cv2
from retinaface import pre_trained_models
from retinaface.predict_single import Model


class FaceDetector():

    def __init__(self, device='cuda:0',
                 output_shape=(256,256),
                 model='retinaface_resnet50_2020-07-20',
                 max_size=250,
                 pad_bbox_ratio=0.0,
                 square_bbox=True,
                 fallback='pass'):

        if model == 'retinaface_resnet50_2020-07-20':
            self.model: Model = pre_trained_models.get_model("resnet50_2020-07-20",
                                                                        max_size=max_size, device=device)
            self.model.eval()
            self.input_size = None
        else:
            raise ValueError('not a correct model')

        self.output_shape = output_shape  # width, height (x,y)
        self.pad_bbox_ratio = pad_bbox_ratio
        self.square_bbox = square_bbox

        self.fallback = fallback
        assert self.fallback in ['pass', 'avg']
        self.bbox_mean = np.zeros(4)
        self.landmark_mean = np.zeros((5,2))
        self.num_tracked = 0


    def get_rgb_image(self, image):
        if isinstance(image, str):
            rgb_img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # nd array is BGR. so change to RGB
            rgb_img = Image.fromarray(image[:, :, ::-1]).convert('RGB')
        elif isinstance(image, Image.Image):
            rgb_img = image.convert('RGB')
        else:
            raise ValueError('not a correct type')
        return rgb_img

    def pad_bbox(self, bbox, padding_ratio, image_shape):
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin
        pad_x = padding_ratio * width
        pad_y = padding_ratio * height
        xmin, ymin, xmax, ymax = xmin-pad_x, ymin-pad_y, xmax+pad_x, ymax+pad_y
        # return (max(xmin, 0), max(ymin, 0), min(image_shape[1], xmax), min(image_shape[0], ymax))
        return xmin, ymin, xmax, ymax

    def make_square_bbox(self, bbox, image_shape):
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin
        if height > width:
            pad1 = (height - width) // 2
            pad2 = (height - width) - pad1
            xmin, xmax = xmin-pad1, xmax+pad2
        else:
            pad1 = (width - height) // 2
            pad2 = (width - height) - pad1
            ymin, ymax = ymin-pad1, ymax+pad2

        # return (max(xmin, 0), max(ymin, 0), min(image_shape[1], xmax), min(image_shape[0], ymax))
        return xmin, ymin, xmax, ymax


    def detect(self, image):
        orig_img = self.get_rgb_image(image)
        rgb_img_np = np.array(orig_img)

        predictions = self.model.predict_jsons(rgb_img_np)

        # selection most confident output
        arg_max_idx = np.argmax([pred['score'] for pred in predictions])
        prediction = predictions[arg_max_idx]

        bbox = prediction['bbox']
        score = prediction['score']
        landmark = np.array(prediction['landmarks'])

        if score == -1:
            # predictions = [{"bbox": [], "score": -1, "landmarks": []}]
            if self.fallback == 'pass':
                success = False
                return success, None, None, None, None, None
            elif self.fallback == 'avg':
                bbox = self.bbox_mean.copy().tolist()
                landmark = self.landmark_mean.copy()
                score = -1
            else:
                raise ValueError('not a correct fallback', self.fallback)

        # if input image has forehead cutoff, bbox is short
        landmark_check = landmark[0] - np.abs(landmark[0] - landmark[3])  # forhead area
        bbox[1] = min(bbox[1], landmark_check[1])

        # pad bbox
        bbox = self.pad_bbox(bbox, padding_ratio=self.pad_bbox_ratio, image_shape=rgb_img_np.shape)
        bbox = [int(np.round(i, 0)) for i in bbox]
        # square bbox
        if self.square_bbox:
            bbox = self.make_square_bbox(bbox, image_shape=rgb_img_np.shape)

        if self.fallback == 'avg':
            # update average
            self.bbox_mean = (self.bbox_mean * self.num_tracked + np.array(bbox)) / (self.num_tracked + 1)
            self.landmark_mean = (self.landmark_mean * self.num_tracked + landmark) / (self.num_tracked + 1)
            self.num_tracked = self.num_tracked + 1

        # crop
        xmin, ymin, xmax, ymax = bbox
        cropped_image = Image.fromarray(rgb_img_np).crop((xmin, ymin, xmax, ymax))
        landmark[:,0] = landmark[:,0] - xmin
        landmark[:,1] = landmark[:,1] - ymin

        cropped_rgb_array = np.array(cropped_image)
        if self.output_shape is not None:
            cropped_rgb_array = cv2.resize(cropped_rgb_array, self.output_shape)
            landmark[:,0] = landmark[:,0] / cropped_image.width * self.output_shape[0]
            landmark[:,1] = landmark[:,1] / cropped_image.height * self.output_shape[1]

        cropped = Image.fromarray(cropped_rgb_array)
        success = True

        return success, orig_img, cropped, bbox, landmark, score
