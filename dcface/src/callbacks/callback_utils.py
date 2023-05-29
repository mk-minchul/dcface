import os
from src.general_utils import os_utils, img_utils
from pytorch_lightning.utilities.distributed import rank_zero_only
import numpy as np
import shutil
import cv2
import pandas as pd
from src.general_utils.os_utils import natural_sort

def make_cache_dir_name(pl_module, stage):
    datamodule_config = pl_module.hparams.datamodule.keywords
    img_size = datamodule_config['img_size']
    datagen_num_subj = datamodule_config['datagen_num_subj']
    datagen_num_img_per_subj = datamodule_config['datagen_num_img_per_subj']
    if 'casia_webface' in datamodule_config['dataset_name']:
        dname = 'casia_webface'
    elif 'ffhq' in datamodule_config['dataset_name']:
        dname = 'ffhq'
    else:
        raise ValueError('not handled yet')
    name = f'stage:{stage}_{dname}_img_size:{img_size}_num_subj:{datagen_num_subj}_num_img_per_subj:{datagen_num_img_per_subj}'
    return name


def make_fullgen_dir_name(data_params):
    img_size = data_params['img_size']
    fullgen_num_train_subject = data_params['fullgen_num_train_subject']
    fullgen_num_extra_subject = data_params['fullgen_num_extra_subject']
    fullgen_mixing_method = data_params['fullgen_mixing_method']
    datagen_num_img_per_subj = data_params['fullgen_num_image_per_subject']
    if 'casia_webface' in data_params['dataset_name']:
        dname = 'casia_webface'
    elif 'ffhq' in data_params['dataset_name']:
        dname = 'ffhq'
    else:
        raise ValueError('not handled yet')
    name = f'stage:fullgen_{dname}_img_size:{img_size}' \
           f'_num_train:{fullgen_num_train_subject}' \
           f'_num_extra:{fullgen_num_extra_subject}' \
           f'_num_img:{datagen_num_img_per_subj}' \
           f'_mixing:{fullgen_mixing_method}'
    # example: stage:fullgen_casia_webface_img_size:112_num_train:10_num_extra:10_num_img:8_mixing:label_interpolate
    return name

def get_label(path):
    return os.path.basename(os.path.dirname(path))

@rank_zero_only
def copy_images_and_clear_cache_dir(pl_module, datagen_save_dir,
                                    keep_subjects=['622', '10322', '10323', '10382', '10489']):

    test_name = make_cache_dir_name(pl_module, stage='test')
    val_name = make_cache_dir_name(pl_module, stage='val')
    val_stage_cache_root = os.path.join(datagen_save_dir,
                                        os.path.basename(pl_module.hparams.paths.output_dir), 'datagen', val_name)
    test_stage_cache_root = os.path.join(datagen_save_dir,
                                         os.path.basename(pl_module.hparams.paths.output_dir), 'datagen', test_name)


    val_stage_images = os_utils.get_all_files(val_stage_cache_root, extension_list=['.png'])
    test_stage_images = os_utils.get_all_files(test_stage_cache_root, extension_list=['.png'])
    train_labels = np.sort(np.unique([get_label(p) for p in val_stage_images if '/train/' in p]))
    eval_labels = np.sort(np.unique([get_label(p) for p in val_stage_images if '/eval/' in p]))
    keep_train_labels = list(train_labels[:5])
    keep_eval_labels = list(eval_labels[:5])
    keep_labels = keep_train_labels + keep_eval_labels + keep_subjects

    val_stage_save_images = list(filter(lambda x: get_label(x) in keep_labels, val_stage_images))
    test_stage_save_images = list(filter(lambda x: get_label(x) in keep_labels, test_stage_images))

    print('length val_stage_save_images', len(val_stage_save_images))
    print('length test_stage_save_images', len(test_stage_save_images))

    # make merged visual and save
    for save_images, name in zip([val_stage_save_images, test_stage_save_images], [val_name, test_name]):
        save_images = pd.DataFrame(save_images, columns=['full_path'])
        save_images['same_label'] = save_images['full_path'].apply(os.path.dirname)
        same_label_image_groups = save_images.groupby('same_label')['full_path'].apply(list)
        for same_label, images in same_label_image_groups.items():
            images = np.sort(images)
            merged_vis = img_utils.stack_images(images, num_cols=len(images), num_rows=1)
            save_name = same_label.split(name+'/')[-1]+'.jpg'
            output_dir = pl_module.hparams.paths.output_dir
            visual_save_dir = os.path.join(output_dir, 'visual', save_name)
            os.makedirs(os.path.dirname(visual_save_dir), exist_ok=True)
            cv2.imwrite(visual_save_dir, merged_vis)

    print('done saving images')

    if os.path.exists(val_stage_cache_root):
        shutil.rmtree(val_stage_cache_root)
    if os.path.exists(test_stage_cache_root):
        shutil.rmtree(test_stage_cache_root)

    print('done cleaning cache dir')

@rank_zero_only
def copy_images_and_clear_fullgen_dir(fullgen_cache_root, pl_module,
                                      keep_subjects=['622', '10322', '10323', '10382', '10489']):

    images = os_utils.get_all_files(fullgen_cache_root, extension_list=['.png'])
    labels = np.array(natural_sort(np.unique([get_label(p) for p in images])))

    train_dataset = pl_module.trainer.datamodule.data_train
    max_label = int(max(train_dataset.label_groups.keys()))
    is_train = np.array([int(str_lab) <= max_label for str_lab in labels])
    train_keep_labels = list(labels[is_train][:10])
    extra_keep_labels = list(labels[~is_train][:10])

    keep_labels = train_keep_labels + extra_keep_labels + keep_subjects
    save_images = list(filter(lambda x: get_label(x) in keep_labels, images))

    print('length val_stage_save_images', len(save_images))

    # make merged visual and save
    save_images = pd.DataFrame(save_images, columns=['full_path'])
    save_images['same_label'] = save_images['full_path'].apply(os.path.dirname)
    same_label_image_groups = save_images.groupby('same_label')['full_path'].apply(list)
    for same_label, images in same_label_image_groups.items():
        images = np.sort(images)
        merged_vis = img_utils.stack_images(images, num_cols=len(images), num_rows=1)
        save_name = f'fullgen/{os.path.basename(same_label)}.jpg'
        output_dir = pl_module.hparams.paths.output_dir
        visual_save_dir = os.path.join(output_dir, 'visual', save_name)
        os.makedirs(os.path.dirname(visual_save_dir), exist_ok=True)
        cv2.imwrite(visual_save_dir, merged_vis)

    print('done saving images')
    data_params = pl_module.hparams.datamodule.keywords
    if os.path.exists(fullgen_cache_root) and data_params['delete_generated_full_dataset']:
        shutil.rmtree(fullgen_cache_root)

    print('done cleaning cache dir')