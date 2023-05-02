# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

# limitations under the License.


import warnings
from typing import Optional, Tuple, Union
from torch.nn import functional as F
import torch

from ...pipeline_utils import DiffusionPipeline, ImagePipelineOutput


class DDIMGuidedPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, unet, scheduler, recognition_model, guidance_scale=1000):
        super().__init__()
        scheduler = scheduler.set_format("pt")
        self.guidance_scale = guidance_scale
        self.register_modules(unet=unet, scheduler=scheduler, recognition_model=recognition_model)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[torch.Generator] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        encoder_hidden_states=None,
        reference_recognition_feature=None,
        return_x0_intermediates=False,
        return_x0_guided_intermediates=False,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            eta (`float`, *optional*, defaults to 0.0):
                The eta parameter which controls the scale of the variance (0 is DDIM and 1 is one type of DDPM).
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipeline_utils.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipeline_utils.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is a list with the
            generated images.
        """

        if "torch_device" in kwargs:
            device = kwargs.pop("torch_device")
            warnings.warn(
                "`torch_device` is deprecated as an input argument to `__call__` and will be removed in v0.3.0."
                " Consider using `pipe.to(torch_device)` instead."
            )

            # Set device as before (to be removed in 0.3.0)
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(device)

        # eta corresponds to η in paper and should be between [0, 1]

        # Sample gaussian noise to begin loop
        image = torch.randn(
            (batch_size, self.unet.in_channels, self.unet.sample_size, self.unet.sample_size),
            generator=generator,
        )
        image = image.to(self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        x0_intermediates = {}
        x0_guided_intermediates = {}
        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(image, t, encoder_hidden_states=encoder_hidden_states).sample
            if return_x0_intermediates:
                x0_pred_vis = self.scheduler.step(model_output, t, image, eta).pred_original_sample
                x0_pred_vis = (x0_pred_vis.clone() / 2 + 0.5).clamp(0, 1)
                x0_pred_vis = x0_pred_vis.cpu().permute(0, 2, 3, 1).numpy()
                x0_intermediates[t] = x0_pred_vis

            # guide
            model_output, image = self.cond_fn(
                image=image,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                model_output=model_output,
                reference_recognition_feature=reference_recognition_feature
            )

            if return_x0_guided_intermediates:
                x0_pred_vis = self.scheduler.step(model_output, t, image, eta).pred_original_sample
                x0_pred_vis = (x0_pred_vis.clone() / 2 + 0.5).clamp(0, 1)
                x0_pred_vis = x0_pred_vis.cpu().permute(0, 2, 3, 1).numpy()
                x0_guided_intermediates[t] = x0_pred_vis

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # do x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, eta).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image, x0_intermediates=x0_intermediates, x0_guided_intermediates=x0_guided_intermediates)

    @torch.enable_grad()
    def cond_fn(
            self,
            image,
            timestep,
            encoder_hidden_states,
            model_output,
            reference_recognition_feature,
    ):
        image = image.detach().requires_grad_()

        latent_model_input = image

        # predict the noise residual
        noise_pred = self.unet(latent_model_input, timestep, encoder_hidden_states=encoder_hidden_states).sample

        alpha_prod_t = self.scheduler.alphas_cumprod[timestep].clone()
        beta_prod_t = 1 - alpha_prod_t
        sample = (image - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
        sample = torch.clip(sample, -1, 1)

        x0_pred_feature, _ = self.recognition_model(sample)
        x0_pred_norm = torch.norm(x0_pred_feature, 2, -1, keepdim=True)
        x0_pred_feature = x0_pred_feature / x0_pred_norm

        loss = spherical_dist_loss(x0_pred_feature, reference_recognition_feature).mean() * self.guidance_scale

        grads = -torch.autograd.grad(loss, image)[0]
        model_output = model_output - torch.sqrt(beta_prod_t) * grads
        return model_output, image

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)
