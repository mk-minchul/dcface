import os
import sys

import torch

from src.dataset import face_dataset
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from src.dataset.make_trainrec import FeatureReader, FeatureSaver, DONE_NAME
from tqdm import tqdm


def version_maker(hparams):
    version = f'img={hparams.img_size}_encoder={hparams.first_stage_config.params.ckpt_path.split("/")[-2]}'
    version = version + f'_orig_augmentations1={hparams.orig_augmentations1}'
    return version


def should_use_record_file(hparams):
    if hparams.record_file_type == 'encoded':
        return True
    return False


def maybe_load_train_rec(image_dataset_path, hparams):
    if should_use_record_file(hparams):
        version = version_maker(hparams)
        feature_saving_root = os.path.join(image_dataset_path, version)
        return FeatureReader(feature_saving_root)
    else:
        return None


def maybe_make_train_rec(image_dataset_path, hparams, pl_module):
    if not should_use_record_file(hparams):
        return None

    print('Preparing Encoded Dataset')
    version = version_maker(hparams)
    feature_saving_root = os.path.join(image_dataset_path, version)
    if os.path.isfile(os.path.join(feature_saving_root, DONE_NAME)):
        return None

    # make train rec
    data_train = face_dataset.make_dataset(image_dataset_path,
                                           deterministic=False,
                                           img_size=hparams.img_size,
                                           return_extra_same_label_samples=hparams.return_extra_same_label_samples,
                                           subset=hparams.train_val_split[0],
                                           orig_augmentations1=hparams.orig_augmentations1,
                                           orig_augmentations2=hparams.orig_augmentations2)

    print(f'Saving at {feature_saving_root}')
    os.makedirs(feature_saving_root, exist_ok=True)
    feature_saver = FeatureSaver(feature_saving_root)

    model_device = pl_module.device
    training = pl_module.training
    pl_module.to('cuda:0')
    pl_module.eval()
    dataloader = DataLoader(dataset=data_train, batch_size=32, num_workers=0, shuffle=False)
    for batch in tqdm(dataloader, total=len(dataloader)):
        with torch.no_grad():
            zs = pl_module.encode_image_step(batch['image']).detach()
        for z in zs:
            feature_saver.feature_encode(z)

    feature_saver.mark_done()

    pl_module.to(model_device)
    if training:
        pl_module.train()

    return None
