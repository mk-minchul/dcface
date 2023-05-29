from pytorch_lightning.callbacks import  Callback
import pytorch_lightning as pl
import torch
import torchvision
import wandb
from src.visualizations import sample_visual
from src.visualizations import train_visualization
import os

def get_dataset(trainer, stage):
    if stage == 'val':
        dataset = trainer.datamodule.data_val
    elif stage == 'test':
        dataset = trainer.datamodule.data_test
    elif stage == 'train':
        dataset = trainer.datamodule.data_train
    else:
        raise ValueError('')
    return dataset

def limit_data_batchsize(batch, batchsize):
    current_batchsize = len(batch['image'])
    new_bs = min(current_batchsize, batchsize)
    new = {}
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            new[key] = val[:new_bs]
        else:
            new[key] = val
    return new, new_bs

class ImageLogger(Callback):
    def __init__(self, num_custom1_visual_subjects=8, num_custom1_visual_images_per_subject=4, check_every_n_epoch=1):
        super().__init__()
        self.num_custom1_visual_subjects = num_custom1_visual_subjects
        self.num_custom1_visual_images_per_subject = num_custom1_visual_images_per_subject
        self.check_every_n_epoch = check_every_n_epoch
        self.perm_order = torch.randperm(num_custom1_visual_subjects * num_custom1_visual_images_per_subject)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        pass

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        pass

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass


    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage='val') -> None:

        print('on_validation_epoch_end image logger')
        if pl_module.global_step == 0 and stage == 'val':
            return

        if pl_module.current_epoch % self.check_every_n_epoch != 0:
            return

        current_step = pl_module.logger.experiment.step if stage == 'test' else pl_module.global_step

        dataset = get_dataset(trainer, stage)
        is_dataset_deterministic = dataset.deterministic
        dataset.deterministic = True

        batch = sample_visual.sample_images_for_vis(dataset,
                                                    self.num_custom1_visual_subjects,
                                                    self.num_custom1_visual_images_per_subject)
        # eval_batch_size = pl_module.hparams['datamodule'].keywords['batch_size_eval']
        # batch, eval_batch_size = limit_data_batchsize(batch, eval_batch_size)

        # DDPM DDIM Sampling
        if pl_module.global_rank == 0:
            # print('DDPM Generating Images')
            # images = sample_visual.render_condition(batch, pl_module=pl_module, sampler='ddpm',
            #                                         between_zero_and_one=True, show_progress=False)
            # grid = torchvision.utils.make_grid(torch.tensor(images.transpose(0, 3, 1, 2)), nrow=4)
            # pl_module.logger.experiment.log({f'{stage}_samples/ddpm_samples': wandb.Image(grid),
            #                                  'epoch': pl_module.current_epoch}, current_step)

            print('DDIM Generating Images')
            images = sample_visual.render_condition(batch, pl_module=pl_module, sampler='ddim',
                                                    between_zero_and_one=True, show_progress=False)
            grid = torchvision.utils.make_grid(torch.tensor(images.transpose(0, 3, 1, 2)),
                                               nrow=self.num_custom1_visual_images_per_subject)
            pl_module.logger.experiment.log({f'{stage}_samples/ddim_samples': wandb.Image(grid),
                                             'epoch': pl_module.current_epoch}, current_step)

            batch['image'] = batch['image'][self.perm_order]
            images = sample_visual.render_condition(batch, pl_module=pl_module, sampler='ddim',
                                                    between_zero_and_one=True, show_progress=False)
            grid = torchvision.utils.make_grid(torch.tensor(images.transpose(0, 3, 1, 2)),
                                               nrow=self.num_custom1_visual_images_per_subject)
            pl_module.logger.experiment.log({f'{stage}_samples/style_perm_image': wandb.Image(grid),
                                             'epoch': pl_module.current_epoch}, current_step)


            print('Extra Sample Visualization')
            if pl_module.hparams.label_mapping['version'] is None and pl_module.hparams.external_mapping['version'] is None:
                pass
            else:
                train_dataset = get_dataset(trainer, 'train')
                is_train_dataset_deterministic = train_dataset.deterministic
                train_dataset.deterministic = True
                output_dir = pl_module.hparams.paths['output_dir']
                extra_vis_save_root = os.path.join(output_dir, f'extra_vis_{stage}', 'epoch_'+str(pl_module.current_epoch))
                train_visualization.visualization_bundle(train_dataset, pl_module, extra_vis_save_root)
                train_dataset.deterministic = is_train_dataset_deterministic

        dataset.deterministic = is_dataset_deterministic

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.on_validation_epoch_end(trainer, pl_module, stage='test')
