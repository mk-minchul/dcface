import torch.nn as nn
import math
import torch
from src.recognition import recognition_helper
import copy


def make_label_mapping(config, unet_config):
    if config.version == None:
        label_mapping = nn.Identity()
    elif config.version == 'v4':
        assert unet_config['params'].condition_type in ['cross_attn', 'crossatt_and_stylemod']
        assert unet_config['params'].condition_source in ['patchstat_spatial_and_image', 'image_and_patchstat_spatial']
        # image condition
        config.recognition_config = copy.copy(config.recognition_config)
        config.recognition_config['ckpt_path'] = None
        config.recognition_config['center_path'] = None
        config.recognition_config['return_spatial'] = [21]
        model = recognition_helper.make_recognition_model(config.recognition_config, enable_training=True)
        label_mapping = ImageEmbedder(backbone=model)
        out = label_mapping.forward(torch.randn(3,3,112,112))
    else:
        raise ValueError('')

    return label_mapping


class ImageEmbedder(nn.Module):

    def __init__(self, backbone, with_cross_attention_adopter=False):
        super(ImageEmbedder, self).__init__()
        self.backbone = backbone
        num_latent = 50
        latent_dim = 512
        pos_emb_init = positionalencoding1d(latent_dim, num_latent) - 0.5
        self.pos_emb = nn.ParameterList([nn.Parameter(pos_emb_init)])
        self.scaler = nn.ParameterList([nn.Parameter(torch.ones(1) * 0.001)])
        if with_cross_attention_adopter:
            self.cross_attn_adapter = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=1)

    def forward(self, x):
        feature, spatial = self.backbone(x)
        spatial = spatial[0]
        shape = spatial.shape
        spatial = spatial.view(shape[0], shape[1], -1)
        feature = feature.unsqueeze(2)
        out = torch.cat([feature, spatial], dim=2).transpose(1, 2)
        out = out + self.pos_emb[0][None, :, :]
        id = out[:, 0, :] * self.scaler[0]
        cross_att = out[:, 1:, :]
        return id, cross_att



def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                          -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


