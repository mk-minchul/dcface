import sys

import torch
from torchvision.utils import make_grid
from src.visualizations import sample_visual
from src.models.conditioner import split_label_spatial
import torchvision
import numpy as np
from src.general_utils.img_utils import prepare_text_img
import os
from tqdm import tqdm
from src.diffusers.pipelines.ddim.pipeline_ddim_cond import DDIMPipeline
import cv2
import imageio


def plot_diffusion(clean_image, pl_module, num_interval=5):

    # clean_image = batch['image'][0:1]
    assert clean_image.ndim == 3
    clean_image = torch.unsqueeze(clean_image, 0)
    noise = torch.randn(clean_image.shape).to(clean_image.device)
    timesteps = torch.tensor(np.linspace(1, 999, num_interval).astype(int), device=clean_image.device ).long()
    clean_image = clean_image.repeat(len(timesteps), 1, 1, 1)
    noise = noise.repeat(len(timesteps), 1, 1, 1)
    diffusion_row = pl_module.noise_scheduler.add_noise(clean_image, noise, timesteps)
    diffusion_grid = make_grid(diffusion_row, nrow=diffusion_row.shape[0])
    no_diffusion_grid = diffusion_grid.cpu().numpy().transpose(1, 2, 0)
    npy_image = sample_visual.to_image_npy_uint8(no_diffusion_grid)

    return npy_image


def visualization_bundle(dataset, pl_module, save_root):
    generate_identity_style_mix_images(dataset, pl_module, num_img_per_subject=4, num_subjects=4,
                                       save_root=os.path.join(save_root, 'exploration1'))
    # generate_identity_style_mix_images(dataset, pl_module, num_img_per_subject=4, num_subjects=8,
    #                                    save_root=os.path.join(save_root, 'exploration2'))

    generate_interpolation(dataset, pl_module, num_img_per_subject=4, num_subjects=4,
                           mixing_method='label_interpolate',
                           save_root=os.path.join(save_root, 'label_interpolation'))

    generate_interpolation(dataset, pl_module, num_img_per_subject=4, num_subjects=4,
                           mixing_method='spatial_interpolate',
                           save_root=os.path.join(save_root, 'spatial_interpolate'))



def generate_identity_style_mix_images(dataset, pl_module, num_img_per_subject, num_subjects, save_root,
                                       sampler='ddim', return_x0_intermediates=False):

    batch = sample_visual.sample_images_for_vis(dataset, num_subjects=num_subjects*2, num_img_per_subject=4)
    label_batch, extra_batch = divide_batch(batch, half=num_img_per_subject*num_subjects)

    generator = torch.manual_seed(0)
    pred_images = sample_visual.render_condition(label_batch, pl_module,
                                                 sampler=sampler, between_zero_and_one=True,
                                                 show_progress=False, generator=generator,
                                                 mixing_batch=extra_batch,
                                                 mixing_method='spatial_interpolate', source_alpha=0.0,
                                                 return_x0_intermediates=return_x0_intermediates)

    if return_x0_intermediates:
        pred_images, x0_intermediates = pred_images
        pred_images_grid = torchvision.utils.make_grid(torch.tensor(pred_images.transpose(0, 3, 1, 2)), nrow=8)
        pred_images_grid_uint8 = sample_visual.to_image_npy_uint8(pred_images_grid.detach().cpu().numpy().transpose(1,2,0))
        sample_visual.save_uint8(pred_images_grid_uint8, path='{}/{}.jpg'.format(save_root, f'all.jpg'))

        for i in range(num_subjects):
            interms = torch.tensor(np.array([val[i] for _, val in x0_intermediates.items()]).transpose(0, 3, 1, 2))
            interms_grid = torchvision.utils.make_grid(interms, nrow=10)
            interms_grid_uint8 = sample_visual.to_image_npy_uint8(interms_grid.detach().cpu().numpy().transpose(1,2,0))
            sample_visual.save_uint8(interms_grid_uint8, path='{}/{}.jpg'.format(save_root, f'interms_{i}.jpg'))

    orig_images = label_batch['image']
    extra_image = extra_batch['image']
    os.makedirs(save_root, exist_ok=True)

    for i in range(num_subjects):
        sub_orig_images = orig_images[i*num_img_per_subject: (i+1)*num_img_per_subject]
        sub_extra_image = extra_image[i*num_img_per_subject: (i+1)*num_img_per_subject]
        sub_pred_images = pred_images[i*num_img_per_subject: (i+1)*num_img_per_subject]

        # visual
        orig_grid = torchvision.utils.make_grid(sub_orig_images * 0.5 + 0.5, nrow=num_img_per_subject)
        orig_grid_uint8 = sample_visual.to_image_npy_uint8(orig_grid.detach().cpu().numpy().transpose(1,2,0))
        orig_text = prepare_text_img('Identity Examples', height=orig_grid_uint8.shape[0], width=240,)
        orig_grid_uint8 = np.concatenate([orig_text, orig_grid_uint8], axis=1)

        extra_grid = torchvision.utils.make_grid(sub_extra_image * 0.5 + 0.5, nrow=num_img_per_subject)
        extra_grid_uint8 = sample_visual.to_image_npy_uint8(extra_grid.detach().cpu().numpy().transpose(1,2,0))
        extra_text = prepare_text_img('Style Images', height=orig_grid_uint8.shape[0], width=240,)
        extra_grid_uint8 = np.concatenate([extra_text, extra_grid_uint8], axis=1)

        grid = torchvision.utils.make_grid(torch.tensor(sub_pred_images.transpose(0, 3, 1, 2)), nrow=num_img_per_subject)
        grid_uint8 = sample_visual.to_image_npy_uint8(grid.detach().cpu().numpy().transpose(1,2,0))
        new_text = prepare_text_img('Generated Samples', height=orig_grid_uint8.shape[0], width=240,)
        grid_uint8 = np.concatenate([new_text, grid_uint8], axis=1)

        vis = np.concatenate([orig_grid_uint8, extra_grid_uint8, grid_uint8], axis=0)
        sample_visual.save_uint8(vis, path='{}/{}.jpg'.format(save_root, i))


def generate_interpolation(dataset, pl_module, num_img_per_subject, num_subjects, mixing_method, save_root,
                           return_x0_intermediates=False):

    batch = sample_visual.sample_images_for_vis(dataset, num_subjects=num_subjects*2, num_img_per_subject=4)
    label_batch, extra_batch = divide_batch(batch, half=num_img_per_subject*num_subjects)

    os.makedirs(save_root, exist_ok=True)

    alphas = np.linspace(1, 0, 10).round(2)
    pred_images_all = []
    for alpha in alphas:
        generator = torch.manual_seed(0)
        pred_images = sample_visual.render_condition(label_batch, pl_module,
                                                     sampler='ddim', between_zero_and_one=True,
                                                     show_progress=False, generator=generator,
                                                     mixing_batch=extra_batch,
                                                     mixing_method=mixing_method, source_alpha=alpha,
                                                     return_x0_intermediates=return_x0_intermediates)

        if return_x0_intermediates:
            pred_images, x0_intermediates = pred_images
            pred_images_grid = torchvision.utils.make_grid(torch.tensor(pred_images.transpose(0, 3, 1, 2)), nrow=8)
            pred_images_grid_uint8 = sample_visual.to_image_npy_uint8(pred_images_grid.detach().cpu().numpy().transpose(1,2,0))
            sample_visual.save_uint8(pred_images_grid_uint8, path='{}/{}.jpg'.format(save_root, f'all_alpha_{alpha:.2f}.jpg'))

            for i in range(num_subjects):
                interms = torch.tensor(np.array([val[i] for _, val in x0_intermediates.items()]).transpose(0, 3, 1, 2))
                interms_grid = torchvision.utils.make_grid(interms, nrow=10)
                interms_grid_uint8 = sample_visual.to_image_npy_uint8(interms_grid.detach().cpu().numpy().transpose(1,2,0))
                sample_visual.save_uint8(interms_grid_uint8, path='{}/{}.jpg'.format(save_root, f'interms_{i}_alpha_{alpha:.2f}.jpg'))

        pred_images_all.append(pred_images)

    orig_images = label_batch['image']
    extra_image = extra_batch['image']
    for i in range(num_subjects):
        sub_orig_images = orig_images[i*num_img_per_subject: (i+1)*num_img_per_subject]
        orig_grid = torchvision.utils.make_grid(sub_orig_images * 0.5 + 0.5, nrow=num_img_per_subject)
        orig_grid_uint8 = sample_visual.to_image_npy_uint8(orig_grid.detach().cpu().numpy().transpose(1,2,0))
        orig_text = prepare_text_img('Subject 1', height=orig_grid_uint8.shape[0], width=340,)
        orig_grid_uint8 = np.concatenate([orig_text, orig_grid_uint8], axis=1)

        sub_extra_image = extra_image[i*num_img_per_subject: (i+1)*num_img_per_subject]
        extra_grid = torchvision.utils.make_grid(sub_extra_image * 0.5 + 0.5, nrow=num_img_per_subject)
        extra_grid_uint8 = sample_visual.to_image_npy_uint8(extra_grid.detach().cpu().numpy().transpose(1,2,0))
        extra_text = prepare_text_img('Subject 2', height=orig_grid_uint8.shape[0], width=340,)
        extra_grid_uint8 = np.concatenate([extra_text, extra_grid_uint8], axis=1)
        vis = [orig_grid_uint8, extra_grid_uint8]

        for alpha, pred_images in zip(alphas, pred_images_all):
            sub_pred_images = pred_images[i*num_img_per_subject: (i+1)*num_img_per_subject]
            grid = torchvision.utils.make_grid(torch.tensor(sub_pred_images.transpose(0, 3, 1, 2)), nrow=num_img_per_subject)
            grid_uint8 = sample_visual.to_image_npy_uint8(grid.detach().cpu().numpy().transpose(1,2,0))
            new_text = prepare_text_img('Generated alpha:{:.2f}'.format(alpha), height=orig_grid_uint8.shape[0], width=340,)
            grid_uint8 = np.concatenate([new_text, grid_uint8], axis=1)
            vis.append(grid_uint8)

        vis = np.concatenate(vis, axis=0)
        sample_visual.save_uint8(vis, path='{}/{}.jpg'.format(save_root, i))




def generate_random_identity_v1(dataset, pl_module, num_img_per_subject, num_subjects, mixing_method, save_root,
                                return_x0_intermediates=False):

    os.makedirs(save_root, exist_ok=True)

    dummy_batch = sample_visual.sample_images_for_vis(dataset, num_subjects=1, num_img_per_subject=1)
    all_label_embs = []
    for label in tqdm(dataset.label_groups.keys(), total=len(dataset.label_groups.keys())):
        tensor_label = torch.tensor([int(label)])
        batch = {}
        batch['image'] = dummy_batch['image']
        batch['class_label'] = tensor_label
        condition_type = pl_module.hparams.unet_config.params.condition_type
        condition_source = pl_module.hparams.unet_config.params.condition_source
        condition = pl_module.get_encoder_hidden_states(batch, batch_size=None)
        label, _ = split_label_spatial(condition_type, condition_source, condition, pl_module=pl_module)
        all_label_embs.append(label.detach().cpu().numpy())

    all_label_embs = np.concatenate(all_label_embs, axis=0)

    # get style
    style_batch = sample_visual.sample_images_for_vis(dataset, num_subjects=num_img_per_subject, num_img_per_subject=1)
    batch = {}
    batch['image'] = style_batch['image']
    batch['class_label'] = torch.tensor([int(0)]*num_img_per_subject)
    condition_type = pl_module.hparams.unet_config.params.condition_type
    condition_source = pl_module.hparams.unet_config.params.condition_source
    condition = pl_module.get_encoder_hidden_states(batch, batch_size=None)
    _, spatial = split_label_spatial(condition_type, condition_source, condition, pl_module=pl_module)

    for i in range(num_subjects):
        select_indices = np.random.choice(len(all_label_embs), 8)
        rand_label_cond = np.stack([all_label_embs[rand_idx, n] for n, rand_idx  in enumerate(select_indices)], axis=0)
        rand_label_cond = torch.tensor(rand_label_cond).unsqueeze(0).repeat(num_img_per_subject, 1, 1).to(spatial.device)
        cross_attn = torch.cat([rand_label_cond, spatial], dim=1)
        encoder_hidden_states = {'cross_attn':cross_attn, 'concat': None, 'add': None, 'center_emb': None}
        generator = torch.manual_seed(0)
        pipeline = DDIMPipeline(
            unet=pl_module.ema_model.averaged_model if pl_module.hparams.use_ema else pl_module.model,
            scheduler=pl_module.noise_scheduler_ddim)
        pipeline.set_progress_bar_config(disable=True)
        pred_result = pipeline(generator=generator, batch_size=num_img_per_subject, output_type="numpy",
                               num_inference_steps=50, eta=1.0, use_clipped_model_output=False,
                               encoder_hidden_states=encoder_hidden_states,
                               return_x0_intermediates=return_x0_intermediates)
        pred_images = pred_result.images
        pred_images = np.clip(pred_images, 0, 1)
        pred_images_grid = torchvision.utils.make_grid(torch.tensor(pred_images.transpose(0, 3, 1, 2)), nrow=8)
        pred_images_grid_uint8 = sample_visual.to_image_npy_uint8(pred_images_grid.detach().cpu().numpy().transpose(1,2,0))
        sample_visual.save_uint8(pred_images_grid_uint8, path='{}/{}.jpg'.format(save_root, f'{i}.jpg'))

    # fix others and change only one column
    select_indices = np.random.choice(len(all_label_embs), 8)
    rand_label_cond = np.stack([all_label_embs[rand_idx, n] for n, rand_idx in enumerate(select_indices)], axis=0)
    for j in range(all_label_embs.shape[1]):
        for n in range(10):
            rand_label_cond[j] = all_label_embs[np.random.choice(len(all_label_embs), 1)[0], j]
            rand_label_cond_tensor = torch.tensor(rand_label_cond.copy()).unsqueeze(0).repeat(num_img_per_subject, 1, 1).to(spatial.device)
            cross_attn = torch.cat([rand_label_cond_tensor, spatial], dim=1)
            encoder_hidden_states = {'cross_attn':cross_attn, 'concat': None, 'add': None, 'center_emb': None}
            generator = torch.manual_seed(0)
            pipeline = DDIMPipeline(
                unet=pl_module.ema_model.averaged_model if pl_module.hparams.use_ema else pl_module.model,
                scheduler=pl_module.noise_scheduler_ddim)
            pipeline.set_progress_bar_config(disable=True)
            pred_result = pipeline(generator=generator, batch_size=num_img_per_subject, output_type="numpy",
                                   num_inference_steps=50, eta=1.0, use_clipped_model_output=False,
                                   encoder_hidden_states=encoder_hidden_states,
                                   return_x0_intermediates=return_x0_intermediates)
            pred_images = pred_result.images
            pred_images = np.clip(pred_images, 0, 1)
            pred_images_grid = torchvision.utils.make_grid(torch.tensor(pred_images.transpose(0, 3, 1, 2)), nrow=8)
            pred_images_grid_uint8 = sample_visual.to_image_npy_uint8(pred_images_grid.detach().cpu().numpy().transpose(1,2,0))
            sample_visual.save_uint8(pred_images_grid_uint8, path=f'{save_root}/freezeothers_{j}_{n}.jpg')



def calculate_avg_emb_and_image(dataset, pl_module, save_path='/mckim/temp/calculate_avg_emb_and_image.pth'):
    if os.path.isfile(save_path):
        saved = torch.load(save_path)
        all_label_embs, average_image = saved['all_label_embs'], saved['average_image']
        return all_label_embs, average_image

    dummy_batch = sample_visual.sample_images_for_vis(dataset, num_subjects=1, num_img_per_subject=1)
    all_label_embs = []
    for label in tqdm(dataset.label_groups.keys(), total=len(dataset.label_groups.keys())):
        tensor_label = torch.tensor([int(label)])
        batch = {}
        batch['image'] = dummy_batch['image']
        batch['class_label'] = tensor_label
        condition_type = pl_module.hparams.unet_config.params.condition_type
        condition_source = pl_module.hparams.unet_config.params.condition_source
        condition = pl_module.get_encoder_hidden_states(batch, batch_size=None)
        label, _ = split_label_spatial(condition_type, condition_source, condition, pl_module=pl_module)
        all_label_embs.append(label.detach().cpu().numpy())
    all_label_embs = np.concatenate(all_label_embs, axis=0)

    average_image = []
    indexes = np.linspace(0, len(dataset)-1, 5000).astype(int)
    for index in tqdm(indexes, total=len(indexes)):
        batch = dataset[index]
        average_image.append(batch['image'])
    average_image = np.stack(average_image, axis=0).mean(axis=0)
    torch.save({'average_image': average_image, 'all_label_embs': all_label_embs}, save_path)
    return all_label_embs, average_image


def generate_random_identity_v2(dataset, pl_module, num_styles, save_root):

    os.makedirs(save_root, exist_ok=True)
    print(save_root)

    all_label_embs, average_image = calculate_avg_emb_and_image(dataset, pl_module, save_path='/mckim/temp/calculate_avg_emb_and_image.pth')

    # get style vectors
    style_batch = sample_visual.sample_images_for_vis(dataset, num_subjects=num_styles, num_img_per_subject=1)
    batch = {}
    batch['image'] = style_batch['image']
    batch['image'][0] = torch.tensor(average_image)
    batch['class_label'] = torch.tensor([int(0)]*num_styles)  # dummy
    condition_type = pl_module.hparams.unet_config.params.condition_type
    condition_source = pl_module.hparams.unet_config.params.condition_source
    condition = pl_module.get_encoder_hidden_states(batch, batch_size=None)
    _, spatial = split_label_spatial(condition_type, condition_source, condition, pl_module=pl_module)

    # save style iamges
    for i, image in enumerate(batch['image']):
        vis_image = image * 0.5 + 0.5
        cv2.imwrite(os.path.join(save_root, f'style_image_{i}.jpg'), vis_image.numpy().transpose(1,2,0)[:,:,::-1]*255)

    # make random identity condition
    select_indices = np.random.choice(len(all_label_embs), 8)
    rand_label_cond = np.stack([all_label_embs[rand_idx, n] for n, rand_idx in enumerate(select_indices)], axis=0)

    # fix others and change only one column
    result = {}
    for column_index in tqdm(range(all_label_embs.shape[1]), total=all_label_embs.shape[1]):
        if column_index not in result:
            result[column_index] = {}
        for n in range(10):
            if n not in result[column_index]:
                result[column_index][n] = {}
            rand_label_cond[column_index] = all_label_embs[np.random.choice(len(all_label_embs), 1)[0], column_index]
            rand_label_cond_tensor = torch.tensor(rand_label_cond.copy()).unsqueeze(0).repeat(num_styles, 1, 1).to(spatial.device)
            cross_attn = torch.cat([rand_label_cond_tensor, spatial], dim=1)
            encoder_hidden_states = {'cross_attn':cross_attn, 'concat': None, 'add': None, 'center_emb': None}
            generator = torch.manual_seed(0)
            pipeline = DDIMPipeline(
                unet=pl_module.ema_model.averaged_model if pl_module.hparams.use_ema else pl_module.model,
                scheduler=pl_module.noise_scheduler_ddim)
            pipeline.set_progress_bar_config(disable=True)
            pred_result = pipeline(generator=generator, batch_size=num_styles, output_type="numpy",
                                   num_inference_steps=50, eta=1.0, use_clipped_model_output=False,
                                   encoder_hidden_states=encoder_hidden_states,
                                   return_x0_intermediates=True)
            intermediates = pred_result.x0_intermediates
            for t, interm in intermediates.items():
                time = t.item()
                if time not in result[column_index][n]:
                    result[column_index][n][time] = {}
                for style_index in range(len(interm)):
                    image = interm[style_index]
                    image = np.clip(image, 0, 1)
                    result[column_index][n][time][style_index] = image

    os.makedirs(save_root, exist_ok=True)
    style_indexes = result[0][0][0].keys()
    times = result[0][0].keys()
    N = result[0].keys()
    column_indexes = result.keys()
    for style_index in style_indexes:
        for column_index in column_indexes:
            save_name = f"style:{style_index}_column:{column_index}.jpg"
            # X: Time
            # Y: N
            rows = []
            for time_idx, time in enumerate(times):
                if time_idx % 5 == 0:
                    row = []
                    for n in N:
                        row.append(result[column_index][n][time][style_index])

                    diff = np.stack(row, axis=0).var(axis=0)
                    diff = (diff - diff.min()) / (diff.max() - diff.min())
                    row = row + [diff]
                    rows.append(row)
            row_length = len(rows[0])
            col_length = len(rows)
            vis = []
            for row_idx in range(row_length):
                vis_row = np.concatenate([rows[col_idx][row_idx] for col_idx in range(col_length)], axis=1)*255
                vis.append(vis_row)
            vis = np.concatenate(vis, axis=0)
            cv2.imwrite(os.path.join(save_root, save_name), vis[:,:,::-1])

            vis = []
            for row_idx in range(row_length):
                vis_row = np.concatenate([rows[col_idx][row_idx] for col_idx in range(col_length)], axis=1)*255
                vis_row = vis_row.astype(np.uint8)
                vis.append(vis_row)

            var_added_plot = []
            for _vis in vis:
                _vis = np.concatenate([vis[-1], _vis], axis=0)
                var_added_plot.append(_vis)
            imageio.mimsave(os.path.join(save_root, save_name.replace('.jpg', '.gif')), var_added_plot,  duration=1)



def divide_batch(batch, half=4):
    label_batch = {}
    extra_batch = {}
    for key, val in batch.items():
        label_batch[key] = val[:half]
        extra_batch[key] = val[half:]
    return label_batch, extra_batch


