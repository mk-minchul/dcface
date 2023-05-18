import subprocess, sys, os
sys.path.append(os.getcwd().split('datagen_framework')[0] + 'datagen_framework')
import os
import torch
from torch import nn as nn
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
from src.recognition.tface_model import Backbone
from src.recognition.adaface import AdaFaceV3
from src.general_utils import os_utils
from src.recognition import tface_model
from functools import partial
from typing import Dict


def disabled_train(mode=True, self=None):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def same_config(config1, config2, skip_keys=[]):
    for key in config1.keys():
        if key in skip_keys:
            pass
        else:
            if config1[key] != config2[key]:
                return False
    return True


def download_ir_pretrained_statedict(backbone_name, dataset_name, loss_fn):

    if backbone_name == 'ir_101' and dataset_name == 'webface4m' and loss_fn == 'adaface':
        root = os_utils.get_project_root(project_name='dcface')
        _name, _id = 'adaface_ir101_webface4m.ckpt', '18jQkqB0avFqWa0Pas52g54xNshUOQJpQ'
    elif backbone_name == 'ir_50' and dataset_name == 'webface4m' and loss_fn == 'adaface':
        root = os_utils.get_project_root(project_name='dcface')
        _name, _id = 'adaface_ir50_webface4m.ckpt', '1BmDRrhPsHSbXcWZoYFPJg2KJn1sd3QpN'
    else:
        raise NotImplementedError()
    checkpoint_path = os.path.join(root, 'pretrained_models', _name)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    if not os.path.isfile(checkpoint_path):
        subprocess.check_call([sys.executable, "-m", "pip", "install", 'gdown'])
        try:
            subprocess.check_call([os.path.expanduser('~/.local/bin/gdown'), '--id', _id])
        except:
            subprocess.check_call([os.path.expanduser('~/anaconda3/envs/pj3/bin/gdown'), '--id', _id])
        if not os.path.isdir(os.path.dirname(checkpoint_path)):
            subprocess.check_call(['mkdir', '-p', os.path.dirname(checkpoint_path)])
        subprocess.check_call(['mv', _name, checkpoint_path])

    assert os.path.isfile(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    model_statedict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
    return model_statedict


def get_dim(style_dims=[]):
    cdim = 0
    for index in style_dims:
        if index == 2:
            cdim += 64
        if index == 4:
            cdim += 128
        if index == 6:
            cdim += 256
        if index == 8:
            cdim += 512
    return cdim

def get_spatial(style_dims=[]):
    spatial_dim = []
    for index in style_dims:
        if index == 2:
            spatial_dim.append((56,56))
        if index == 4:
            spatial_dim.append((28,28))
        if index == 6:
            spatial_dim.append((14,14))
        if index == 8:
            spatial_dim.append((7,7))
    return spatial_dim

dict_name_to_filter = {
    "PIL": {
        "bicubic": Image.BICUBIC,
        "bilinear": Image.BILINEAR,
        "nearest": Image.NEAREST,
        "lanczos": Image.LANCZOS,
        "box": Image.BOX
    }
}


def resize_images(x, resizer, ToTensor, mean, std, device):
    x = x.transpose((0, 2, 3, 1))
    x = list(map(lambda x: ToTensor(resizer(x)), list(x)))
    x = torch.stack(x, 0).to(device)
    x = (x/255.0 - mean)/std
    return x


def make_resizer(library, filter, output_size):
    if library == "PIL":
        s1, s2 = output_size
        def resize_single_channel(x_np):
            img = Image.fromarray(x_np.astype(np.float32), mode='F')
            img = img.resize(output_size, resample=dict_name_to_filter[library][filter])
            return np.asarray(img).reshape(s1, s2, 1)
        def func(x):
            x = [resize_single_channel(x[:, :, idx]) for idx in range(3)]
            x = np.concatenate(x, axis=2).astype(np.float32)
            return x
    elif library == "PyTorch":
        import warnings
        # ignore the numpy warnings
        warnings.filterwarnings("ignore")
        def func(x):
            x = torch.Tensor(x.transpose((2, 0, 1)))[None, ...]
            x = F.interpolate(x, size=output_size, mode=filter, align_corners=False)
            x = x[0, ...].cpu().data.numpy().transpose((1, 2, 0)).clip(0, 255)
            return x
    else:
        raise NotImplementedError('library [%s] is not include' % library)
    return func



def return_head(head_name='adaface', class_num=205990, head_m=0.4):
    if head_name == 'adaface':
        head = AdaFaceV3(embedding_size=512,
                         classnum=class_num,
                         m=head_m,
                         scaler_fn='batchnorm',
                         rad_h=-0.333,
                         s=64.0,
                         t_alpha=0.01,
                         cut_gradient=True,
                         head_b=0.4)
    elif head_name == '' or head_name == 'none':
        return None
    else:
        raise ValueError('not implemented yet')
    return head



class RecognitionModel(nn.Module):

    def __init__(self, backbone: Backbone, head: AdaFaceV3, recognition_config: Dict, center: nn.Embedding):
        super(RecognitionModel, self).__init__()
        self.backbone = backbone
        self.recognition_config = recognition_config

        self.size = 112
        self.resizer = make_resizer("PIL", "bilinear", (self.size, self.size))
        self.totensor = transforms.ToTensor()
        self.mean = torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        self.std = torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        self.swap_channel = True
        self.head = head
        self.center = center


    def forward(self, x, orig_images=None):

        if orig_images is not None and orig_images.shape[2] == self.size:
            x = orig_images

        elif x.shape[2] != self.size or x.shape[3] != self.size:
            print('why is this happening?')
            quantized_x = self.quantize_images(x)
            x = self.resize_and_normalize(quantized_x, device=x.device)

        if self.swap_channel:
            x = torch.flip(x, dims=[1])

        # from general_utils import img_utils
        # import cv2
        # cv2.imwrite('/mckim/temp/temp.png', img_utils.tensor_to_numpy(x[0].cpu()))
        feature, norm, spatials = self.backbone(x, return_spatial=self.recognition_config.return_spatial)

        if self.recognition_config.normalize_feature:
            feature = feature
        else:
            feature = feature * norm

        if self.recognition_config.return_spatial:
            return feature, spatials
        else:
            return feature

    def classify(self, features, norms, label):
        return self.head(features, norms, label)

    def feature_normalize(self, input_z):
        input_norm = torch.norm(input_z, 2, -1, keepdim=True)
        input_z = input_z / input_norm
        return input_z, input_norm

    def ce_loss(self, input, label):
        input_z, _ = self.forward(input)
        input_norm = torch.norm(input_z, 2, -1, keepdim=True)
        input_z = input_z / input_norm
        input_z_logits, _ = self.classify(input_z, input_norm, label)
        loss_ce = F.cross_entropy(input_z_logits, label)
        return loss_ce

    def quantize_images(self, x):
        if self.mean.device != x.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)
        x = (x * self.std) + self.mean
        x = (255.0 * x + 0.5).clamp(0.0, 255.0)
        x = x.detach().cpu().numpy().astype(np.uint8)
        return x

    def resize_and_normalize(self, x, device):
        out = resize_images(x, resizer=self.resizer, ToTensor=self.totensor,
                            mean=self.mean, std=self.std, device=device)
        return out


def make_recognition_model(recognition_config, enable_training=False):

    if not recognition_config:
        return None

    if 'ir_101' == recognition_config.backbone:
        print('making IR_101')
        backbone = tface_model.IR_101(input_size=(112, 112))
        backbone_name = 'ir_101'
    elif 'ir_50' == recognition_config.backbone:
        print('making IR_50')
        backbone_name = 'ir_50'
        backbone = tface_model.IR_50(input_size=(112, 112))
    else:
        raise NotImplementedError()

    head = return_head(head_name=recognition_config.head_name)

    if recognition_config.ckpt_path:
        print('loading backbone and head checkpoint from ')
        print(recognition_config.ckpt_path)
        statedict = torch.load(recognition_config.ckpt_path, map_location='cpu')['state_dict']
        backbone.load_state_dict({k.replace("model.", ''): v for k, v in statedict.items() if 'model.' in k})
        if head is not None:
            head.load_state_dict({k.replace("head.", ''): v for k, v in statedict.items() if 'head.' in k})
    else:
        # load statedict
        assert recognition_config.dataset == 'webface4m'
        assert recognition_config.loss_fn == 'adaface'
        print('Loading pretrained IR model trained with adaface webface4m')
        model_statedict = download_ir_pretrained_statedict(backbone_name, 'webface4m', 'adaface')
        backbone.load_state_dict(model_statedict, strict=True)

    if recognition_config.center_path:
        print('Loading precomputed center', recognition_config.center_path)
        center = torch.load(recognition_config.center_path, map_location='cpu')['center']
        center_emb = nn.Embedding(num_embeddings=center.shape[0], embedding_dim=center.shape[1])
        center_emb.load_state_dict({'weight': center}, strict=True)
    else:
        center_emb = None

    model = RecognitionModel(backbone=backbone, head=head, recognition_config=recognition_config, center=center_emb)
    if enable_training:
        print('enable training')
        pass
    else:
        model = model.eval()
        model.train = partial(disabled_train, self=model)
        for param in model.parameters():
            param.requires_grad = False

    return model

