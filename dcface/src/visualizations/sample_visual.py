import torch
import numpy as np
from PIL import Image
import os
from src.diffusers.pipelines.ddpm.pipeline_ddpm_cond import DDPMPipeline
from src.diffusers.pipelines.ddim.pipeline_ddim_cond import DDIMPipeline
from src.diffusers.pipelines.ddim.pipeline_ddim_guide import DDIMGuidedPipeline
from src.models.conditioner import mix_hidden_states
from src.visualizations.resizer import Resizer


def sample_index_for_visualization(dataset, num_subjects, num_img_per_subject):

    record_info = dataset.record_info.loc[range(dataset.start_index, dataset.start_index+len(dataset))].copy()
    record_info['target'] = record_info['label'].apply(lambda x:dataset.rec_label_to_another_label[x])
    record_info['dataset_index'] = record_info.index - dataset.start_index
    record_info.set_index('dataset_index', inplace=True)
    record_info['dataset_index'] = record_info.index
    groupby = record_info.groupby('target')['dataset_index']
    num_images_per_target = groupby.apply(len)
    valid_num_images_per_target = num_images_per_target[num_images_per_target >= num_img_per_subject]
    valid_num_images_per_target = valid_num_images_per_target.sort_values(ascending=False)
    index_spacing = np.linspace(0, (len(valid_num_images_per_target)-1)//2, num_subjects).astype(int)
    target_selected = valid_num_images_per_target.index[index_spacing]
    per_target_index = groupby.apply(list)
    sample_index = []
    sample_labels = []
    for target in target_selected:
        valid_index = np.array(per_target_index[target])
        index_spacing = np.linspace(0, len(valid_index)-1, num_img_per_subject).astype(int)
        sample_index.extend(list(valid_index[index_spacing]))
        sample_labels.extend([target] * len(index_spacing))

    return sample_index, sample_labels


def sample_images_for_vis(dataset, num_subjects, num_img_per_subject, ):
    sample_index, sample_labels = sample_index_for_visualization(dataset, num_subjects, num_img_per_subject)
    imgs = []
    orig_images = []
    id_images = []
    for i in sample_index:
        data = dataset[i]
        imgs.append(data['image'])
        orig_images.append(data['orig'])
        if 'id_image' in data:
            id_images.append(data['id_image'])

    imgs = torch.stack(imgs, dim=0)
    orig_imgs = torch.stack(orig_images, dim=0)
    id_images = torch.stack(id_images, dim=0)
    sample_labels = torch.tensor(sample_labels)
    sample_index = torch.tensor(sample_index)
    batch = {'image': imgs, 'class_label': sample_labels, 'index': sample_index, 'orig': orig_imgs}
    if len(id_images) > 0:
        batch['id_image'] = id_images
    return batch


@torch.no_grad()
def render_condition(batch, pl_module, sampler='ddim', between_zero_and_one=True, show_progress=False,
                     generator=None, mixing_batch=None, mixing_method='label_interpolate', source_alpha=0.0,
                     return_x0_intermediates=False):
    if generator is None:
        generator = torch.manual_seed(0)

    batch_size = len(batch['image'])
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            batch[key] = val.to(pl_module.device)

    encoder_hidden_states = pl_module.get_encoder_hidden_states(batch, batch_size)
    if mixing_batch is not None:
        for key, val in mixing_batch.items():
            if isinstance(val, torch.Tensor):
                mixing_batch[key] = val.to(pl_module.device)
        mixing_hidden_states = pl_module.get_encoder_hidden_states(mixing_batch, batch_size)
        encoder_hidden_states = mix_hidden_states(encoder_hidden_states, mixing_hidden_states,
                                                  condition_type=pl_module.hparams.unet_config.params.condition_type,
                                                  condition_source=pl_module.hparams.unet_config.params.condition_source,
                                                  mixing_method=mixing_method,
                                                  source_alpha=source_alpha,
                                                  pl_module=pl_module)
    if sampler == 'ddpm':
        pipeline = DDPMPipeline(
            unet=pl_module.ema_model.averaged_model if pl_module.hparams.use_ema else pl_module.model,
            scheduler=pl_module.noise_scheduler)
        pipeline.set_progress_bar_config(disable=not show_progress)
        pred_result = pipeline(generator=generator, batch_size=batch_size, output_type="numpy",
                               encoder_hidden_states=encoder_hidden_states)
    elif sampler == 'ddim':
        pipeline = DDIMPipeline(
            unet=pl_module.ema_model.averaged_model if pl_module.hparams.use_ema else pl_module.model,
            scheduler=pl_module.noise_scheduler_ddim)
        pipeline.set_progress_bar_config(disable=not show_progress)
        pred_result = pipeline(generator=generator, batch_size=batch_size, output_type="numpy",
                               num_inference_steps=50, eta=1.0, use_clipped_model_output=False,
                               encoder_hidden_states=encoder_hidden_states,
                               return_x0_intermediates=return_x0_intermediates)
    elif sampler == 'ddim_ilvr':
        pipeline = DDIMPipeline(
            unet=pl_module.ema_model.averaged_model if pl_module.hparams.use_ema else pl_module.model,
            scheduler=pl_module.noise_scheduler_ddim)
        pipeline.set_progress_bar_config(disable=not show_progress)
        down_N = 8
        range_t = 100
        shape = batch['image'].shape
        shape_d = (shape[0], 3, int(shape[2] /down_N), int(shape[3] /down_N))
        down = Resizer(shape, 1 / down_N).to(pl_module.device)
        up = Resizer(shape_d, down_N).to(pl_module.device)
        ilvr_params = [down, up, range_t, batch['image']]
        pred_result = pipeline(generator=generator, batch_size=batch_size, output_type="numpy",
                               num_inference_steps=50, eta=1.0, use_clipped_model_output=False,
                               encoder_hidden_states=encoder_hidden_states,
                               return_x0_intermediates=return_x0_intermediates,
                               ilvr=ilvr_params)
    elif sampler == 'ddim_guided':
        pl_module.recognition_model.device = pl_module.device
        pipeline = DDIMGuidedPipeline(
            unet=pl_module.ema_model.averaged_model if pl_module.hparams.use_ema else pl_module.model,
            recognition_model=pl_module.recognition_model,
            scheduler=pl_module.noise_scheduler_ddim)
        pipeline.set_progress_bar_config(disable=not show_progress)
        reference_recognition_feature = encoder_hidden_states['center_emb']
        pred_result = pipeline(generator=generator, batch_size=batch_size, output_type="numpy",
                               num_inference_steps=50, eta=1.0, use_clipped_model_output=False,
                               encoder_hidden_states=encoder_hidden_states,
                               reference_recognition_feature=reference_recognition_feature,
                               return_x0_intermediates=return_x0_intermediates)
    else:
        raise ValueError('')
    pred_images = pred_result.images
    pred_images = np.clip(pred_images, 0, 1)
    if not between_zero_and_one:
        # between -1 and 1
        pred_images = (pred_images - 0.5) / 0.5

    if return_x0_intermediates:
        x0_intermediates = pred_result.x0_intermediates
        return pred_images, x0_intermediates

    return pred_images


def to_image_npy_uint8(pred_images_npy):
    pred_images = np.clip(pred_images_npy*255+0.5, 0, 255).astype(np.uint8)
    return pred_images


def save_uint8(pred_uint8_image, path):
    im = Image.fromarray(pred_uint8_image)
    im.save(path)
