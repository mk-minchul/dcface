import numpy as np
import torch

def make_condition(pl_module, condition_type, condition_source, batch):
    if condition_type is None:
        return None

    result = {'cross_attn': None, 'concat': None, 'add': None, 'center_emb': None}

    if condition_type == 'cross_attn' and condition_source == 'label_center':
        assert 'class_label' in batch
        class_label = batch['class_label'].to(pl_module.device)
        center_emb = pl_module.recognition_model.center(class_label).unsqueeze(1)
        label_emb = pl_module.label_mapping(center_emb)
        result['cross_attn'] = label_emb

    elif condition_type == 'cross_attn' and condition_source == 'spatial_and_label_center':
        assert 'class_label' in batch
        class_label = batch['class_label'].to(pl_module.device)
        center_emb = pl_module.recognition_model.center(class_label).unsqueeze(1)
        _, spatial = pl_module.recognition_model(batch['image'].to(pl_module.device))
        label_emb = pl_module.label_mapping(center_emb)
        ext_mapping = pl_module.external_mapping(spatial).unsqueeze(1).repeat(1, label_emb.shape[1], 1)
        result['cross_attn'] = torch.cat([label_emb, ext_mapping], dim=2)

    elif condition_type == 'add_and_cat' and condition_source == 'spatial_and_label_center':
        assert 'class_label' in batch
        class_label = batch['class_label'].to(pl_module.device)
        center_emb = pl_module.recognition_model.center(class_label)
        _, spatial = pl_module.recognition_model(batch['image'].to(pl_module.device))
        label_emb = pl_module.label_mapping(center_emb)
        result['add'] = label_emb
        ext_mapping = pl_module.external_mapping(spatial)
        result['concat'] = ext_mapping

    elif condition_type == 'cross_attn' and condition_source == 'patchstat_spatial_and_linear_label_center':
        assert 'class_label' in batch
        class_label = batch['class_label'].to(pl_module.device)
        center_emb = pl_module.recognition_model.center(class_label).unsqueeze(1)
        _, spatial = pl_module.recognition_model(batch['image'].to(pl_module.device))
        label_emb = pl_module.label_mapping(center_emb)
        if isinstance(label_emb, list):
            label_emb, vqloss = label_emb
            if pl_module.training:
                result['vqloss'] = vqloss
        ext_mapping = pl_module.external_mapping(spatial)
        result['cross_attn'] = torch.cat([label_emb, ext_mapping], dim=1)

    elif condition_type == 'crossatt_and_stylemod' and condition_source == 'patchstat_spatial_and_image':
        assert 'id_image' in batch
        id_feat, id_cross_att = pl_module.label_mapping(batch['id_image'])
        _, spatial = pl_module.recognition_model(batch['image'].to(pl_module.device))
        ext_mapping = pl_module.external_mapping(spatial)
        cross_attn = torch.cat([id_cross_att, ext_mapping], dim=1).transpose(1,2)
        result['cross_attn'] = pl_module.external_mapping.cross_attn_adapter(cross_attn)
        result['stylemod'] = id_feat

        class_label = batch['class_label'].to(pl_module.device)
        center_emb = pl_module.recognition_model.center(class_label).unsqueeze(1)
    elif condition_type == 'crossatt_and_stylemod' and condition_source == 'image_and_patchstat_spatial':
        assert 'id_image' in batch
        id_feat, id_cross_att = pl_module.label_mapping(batch['id_image'])
        id_cross_att = torch.cat([id_feat.unsqueeze(1), id_cross_att], dim=1)
        _, spatial = pl_module.recognition_model(batch['image'].to(pl_module.device))
        ext_mapping = pl_module.external_mapping(spatial)
        cross_attn = id_cross_att.transpose(1,2)
        result['cross_attn'] = pl_module.label_mapping.cross_attn_adapter(cross_attn)
        result['stylemod'] = ext_mapping

        class_label = batch['class_label'].to(pl_module.device)
        center_emb = pl_module.recognition_model.center(class_label).unsqueeze(1)
    else:
        raise ValueError('make Condition')

    if center_emb.shape[1] == 1:
        center_emb = center_emb.squeeze(1)

    result['center_emb'] = center_emb
    return result


def mix_hidden_states(encoder_hidden_states, mixing_hidden_states, condition_type, condition_source,
                      mixing_method='label_interpolate', source_alpha=0.0, pl_module=None):

    result = {'cross_attn': None, 'concat': None, 'add': None, 'center_emb': None}

    source_label, source_spatial = split_label_spatial(condition_type, condition_source, encoder_hidden_states, pl_module)
    mixing_label, mixing_spatial = split_label_spatial(condition_type, condition_source, mixing_hidden_states, pl_module)

    if condition_type == 'cross_attn' and condition_source == 'label_center':
        assert mixing_method == 'label_interpolate'
        mixed = source_alpha * source_label + (1-source_alpha) * mixing_label
        result['cross_attn'] = mixed

    elif condition_type == 'cross_attn' and condition_source == 'spatial_and_label_center':
        if mixing_method == 'label_interpolate' or mixing_method == 'label_extrapolate':
            mixed_label = source_alpha * source_label + (1-source_alpha) * mixing_label
            mixed = torch.cat([mixed_label, source_spatial], dim=2)
        elif mixing_method == 'spatial_interpolate':
            mixed_spatial = source_alpha * source_spatial + (1-source_alpha) * mixing_spatial
            mixed = torch.cat([source_label, mixed_spatial], dim=2)
        assert mixed.shape == encoder_hidden_states['cross_attn'].shape
        result['cross_attn'] = mixed

    elif condition_type == 'add_and_cat' and condition_source == 'spatial_and_label_center':
        if mixing_method == 'label_interpolate' or mixing_method == 'label_extrapolate':
            mixed_label = source_alpha * source_label + (1-source_alpha) * mixing_label
            result['add'] = mixed_label
            result['concat'] = source_spatial
        elif mixing_method == 'spatial_interpolate':
            mixed_spatial = source_alpha * source_spatial + (1-source_alpha) * mixing_spatial
            result['concat'] = mixed_spatial
            result['add'] = source_label

    elif condition_type == 'cross_attn' and condition_source == 'patchstat_spatial_and_linear_label_center':
        num_label_features = pl_module.label_mapping.pos_emb[0].shape[0]
        if mixing_method == 'label_interpolate':
            mixed_label = source_alpha * source_label + (1-source_alpha) * mixing_label
            mixed = torch.cat([mixed_label, source_spatial], dim=1)
        elif mixing_method == 'label_extrapolate':
            source_num_channel = int(num_label_features * source_alpha)
            mixed_label = torch.cat([source_label[:, :source_num_channel, :],
                                     mixing_label[:, source_num_channel:, :]], dim=1)
            mixed = torch.cat([mixed_label, source_spatial], dim=1)
        elif mixing_method == 'label_extrapolate_random_channel':
            if source_alpha is None:
                # between 1 and 8-1(7)
                source_num_channel = np.random.randint(1, num_label_features)
            elif isinstance(source_alpha, int):
                source_num_channel = int(source_alpha)
                assert source_num_channel <= num_label_features
            else:
                raise ValueError('not accepting float alpha for sanity check')
            source_channels = np.random.choice(range(num_label_features), source_num_channel, replace=False)
            source_channels = np.sort(source_channels)
            mixed_label = torch.zeros_like(source_label)
            for channel in range(num_label_features):
                if channel in source_channels:
                    mixed_label[:, channel, :] = mixed_label[:, channel, :] + source_label[:, channel, :]
                else:
                    mixed_label[:, channel, :] = mixed_label[:, channel, :] + mixing_label[:, channel, :]
            mixed = torch.cat([mixed_label, source_spatial], dim=1)
            result['source_channels'] = source_channels
        elif mixing_method == 'spatial_interpolate':
            mixed_spatial = source_alpha * source_spatial + (1-source_alpha) * mixing_spatial
            mixed = torch.cat([source_label, mixed_spatial], dim=1)
        else:
            raise ValueError('')
        assert mixed.shape == encoder_hidden_states['cross_attn'].shape
        result['cross_attn'] = mixed
    elif condition_type == 'crossatt_and_stylemod' and condition_source == 'patchstat_spatial_and_image':
        source_label_feat, source_label_spat = source_label
        mixing_label_feat, mixing_label_spat = mixing_label

        if mixing_method == 'label_interpolate' or mixing_method == 'label_extrapolate':
            mixed_label_spat = source_alpha * source_label_spat + (1-source_alpha) * mixing_label_spat
            mixed_label_feat = source_alpha * source_label_feat + (1-source_alpha) * mixing_label_feat
            mixed_cross_attn = torch.cat([mixed_label_spat, source_spatial], dim=2)
            mixed_stylemod = mixed_label_feat
        elif mixing_method == 'spatial_interpolate':
            mixed_spatial = source_alpha * source_spatial + (1-source_alpha) * mixing_spatial
            mixed_cross_attn = torch.cat([source_label_spat, mixed_spatial], dim=2)
            mixed_stylemod = source_label_feat

        result['cross_attn'] = mixed_cross_attn
        result['stylemod'] = mixed_stylemod

    elif condition_type == 'crossatt_and_stylemod' and condition_source == 'image_and_patchstat_spatial':
        if mixing_method in ['label_interpolate', 'label_extrapolate']:
            mixed_label = source_alpha * source_label + (1-source_alpha) * mixing_label
            mixed_cross_attn = mixed_label
            mixed_stylemod = source_spatial
        elif mixing_method == 'spatial_interpolate':
            mixed_spatial = source_alpha * source_spatial + (1-source_alpha) * mixing_spatial
            mixed_cross_attn = source_label
            mixed_stylemod = mixed_spatial
        result['cross_attn'] = mixed_cross_attn
        result['stylemod'] = mixed_stylemod
    else:
        raise ValueError('not implemented yet')

    source_class_emb = encoder_hidden_states['center_emb']
    mixing_class_emb = mixing_hidden_states['center_emb']
    if 'spatial' in mixing_method:
        mixed_class_emb = source_class_emb
    else:
        mixed_class_emb = source_alpha * source_class_emb + (1-source_alpha) * mixing_class_emb
        mixed_class_emb = mixed_class_emb / torch.norm(mixed_class_emb, 2, -1, keepdim=True)
    result['center_emb'] = mixed_class_emb
    return result


def split_label_spatial(condition_type, condition_source, encoder_hidden_states, pl_module=None):

    if condition_type == 'cross_attn' and condition_source == 'label_center':
        label = encoder_hidden_states['cross_attn']
        spatial = None

    elif condition_type == 'cross_attn' and condition_source == 'spatial_and_label_center':
        label_dim = 512
        label = encoder_hidden_states['cross_attn'][:,:,:label_dim]
        spatial = encoder_hidden_states['cross_attn'][:,:,label_dim:]

    elif condition_type == 'add_and_cat' and condition_source == 'spatial_and_label_center':
        label = encoder_hidden_states['add']
        spatial = encoder_hidden_states['concat']

    elif condition_type == 'cross_attn' and condition_source == 'patchstat_spatial_and_linear_label_center':
        num_label_features = pl_module.label_mapping.pos_emb[0].shape[0]
        label = encoder_hidden_states['cross_attn'][:, :num_label_features, :]
        spatial = encoder_hidden_states['cross_attn'][:, num_label_features:, :]
    elif condition_type == 'crossatt_and_stylemod' and condition_source == 'patchstat_spatial_and_image':
        num_label_features = pl_module.label_mapping.pos_emb[0].shape[0] - 1
        label_feat = encoder_hidden_states['stylemod']
        label_spat = encoder_hidden_states['cross_attn'][:, :, :num_label_features]
        spatial = encoder_hidden_states['cross_attn'][:, :, num_label_features:]
        label = [label_feat, label_spat]
    elif condition_type == 'crossatt_and_stylemod' and condition_source == 'image_and_patchstat_spatial':
        spatial = encoder_hidden_states['stylemod']
        label = encoder_hidden_states['cross_attn']
    else:
        raise ValueError('not implemented')

    return label, spatial
