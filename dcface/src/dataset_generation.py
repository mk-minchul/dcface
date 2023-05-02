import pandas as pd
import pyrootutils
import dotenv
import os
import torch
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
dotenv.load_dotenv(dotenv_path=root.parent.parent / '.env', override=True)
assert os.getenv('DATA_ROOT')
assert os.path.isdir(os.getenv('DATA_ROOT'))

LOG_ROOT = str(root.parent.parent / 'experiments')
os.environ.update({'LOG_ROOT': LOG_ROOT})
os.environ.update({'PROJECT_TASK': root.stem})
os.environ.update({'REPO_ROOT': str(root.parent.parent)})

import hydra
from pytorch_lightning import LightningModule, LightningDataModule
from argparse import ArgumentParser
import omegaconf
import functools
from src.visualizations.extra_visualization import dataset_generate_mimic_train
from src.general_utils.os_utils import get_all_files
from src.visualizations.extra_visualization import ListDatasetWithIndex
import numpy as np
from src.visualizations.record import Writer


def replace_path(item):
    if isinstance(item, dict) or isinstance(item, omegaconf.dictconfig.DictConfig):
        for key, val in item.items():
            if isinstance(val, str) and os.getcwd().startswith('/mckim'):
                if 'facerec_framework' in val:
                    item[key] = os.path.join('/mckim/MSU/face_rec/facerec_framework',
                                             val.split('facerec_framework')[1][1:])
                if '/mnt/home/kimminc2/data' in val:
                    item[key] = os.path.join('/data/data/faces',
                                             val.split('/mnt/home/kimminc2/data')[1][1:])
                if '/mnt/home/kimminc2/scratch/data' in val:
                    item[key] = os.path.join('/data/data/faces',
                                             val.split('/mnt/home/kimminc2/scratch/data')[1][1:])
                if 'datagen_framework_v2' in val:
                    item[key] = os.path.join('/mckim/MSU/datagen/datagen_framework_v2',
                                             val.split('datagen_framework_v2')[1][1:])
            elif isinstance(val, str) and os.getcwd().startswith('/mnt'):
                print('hpcc')
                if 'facerec_framework' in val:
                    item[key] = os.path.join('/mnt/ufs18/home-150/kimminc2/projects/faces/facerec_framework',
                                             val.split('facerec_framework')[1][1:])
                if 'datagen_framework_v2' in val:
                    item[key] = os.path.join('/mnt/ufs18/home-150/kimminc2/projects/datagen/datagen_framework_v2',
                                             val.split('datagen_framework_v2')[1][1:])
                if '/data/data/faces' in val:
                    item[key] = val.replace('/data/data/faces', '/mnt/home/kimminc2/data')
            elif isinstance(val, functools.partial):
                item[key] = functools.partial(val.func, keywords=replace_path(val.keywords))
            elif isinstance(val, dict) or isinstance(val, omegaconf.dictconfig.DictConfig):
                item[key] = replace_path(val)
            else:
                pass

    return item


def main():

    parser = ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='experiments/datadm_022/e:10_patchstat_spatial_and_image_v4_dropout:0.3_id:0.05_polynomial_1_mix_trim_outlier_10-27_0/checkpoints/epoch_009.ckpt')
    parser.add_argument('--num_image_per_subject', type=int, default=50)
    parser.add_argument('--num_subject', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_partition', type=int, default=1)
    parser.add_argument('--partition_idx', type=int, default=0)

    parser.add_argument('--save_root', type=str, default=None)
    parser.add_argument('--id_images_root', default='/data/data/faces/datagen/id_images/same_gender_same_race_random_threshold:0.3/same_gender_same_race_random.csv')
    parser.add_argument('--style_images_root', type=str, default='train')

    parser.add_argument('--style_sampling_method', type=str, default='train_data',
                        choices=['random', 'same_gender_same_race', 'train_data'])
    parser.add_argument('--use_writer', action='store_true')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    ckpt = torch.load(os.path.join(project_root, args.ckpt_path))
    model_hparam = ckpt['hyper_parameters']
    model_hparam['_target_'] = 'src.diffusion.trainer.Trainer'
    model_hparam['_partial_'] = True

    model_hparam = replace_path(model_hparam)
    print('Paths:')
    print(omegaconf.OmegaConf.to_yaml(model_hparam['paths']))

    # load pl_module
    if 'earth_scratch_lab:v2_ext:v3_cross_attn_id_loss:0.1_hindge_0.3_10-07_0' in args.ckpt_path:
        model_hparam['external_mapping']['spatial_dim'] = 2
    if 'freeze_unet' not in model_hparam['unet_config']:
        model_hparam['unet_config']['freeze_unet'] = False
    pl_module: LightningModule = hydra.utils.instantiate(model_hparam)()
    print('Instantiated ', model_hparam['_target_'])
    pl_module.load_state_dict(ckpt['state_dict'], strict=True)
    pl_module.to('cuda')
    pl_module.eval()

    # load style dataset (training data
    if args.style_images_root == 'train':
        dataconfig = omegaconf.OmegaConf.create(model_hparam['datamodule'].keywords['keywords'])
        dataconfig['_target_'] = 'src.datamodules.face_datamodule.FaceDataModule'
        dataconfig['_partial'] = True
        dataconfig['train_val_split'] = ['0-all', '0.95-1.0']
        datamodule: LightningDataModule = hydra.utils.instantiate(dataconfig)
        datamodule.setup()
        style_dataset = datamodule.data_train
        style_dataset.deterministic = True
    else:
        style_images = get_all_files(args.style_images_root, extension_list=['.png', '.jpg'])
        style_dataset = ListDatasetWithIndex(style_images, flip_color=True)

    # load id images
    if not args.id_images_root.startswith('/'):
        args.id_images_root = os.path.join(pl_module.hparams.paths.data_dir, args.id_images_root)
    print('id_images_root', args.id_images_root)
    if os.path.isdir(args.id_images_root):
        id_images = get_all_files(args.id_images_root, extension_list=['.png', '.jpg'])
        np.random.shuffle(id_images)
        id_dataset = ListDatasetWithIndex(id_images, flip_color=True)
    elif os.path.isfile(args.id_images_root):
        id_images = pd.read_csv(args.id_images_root)['path'].tolist()
        id_images = [os.path.join(os.path.dirname(args.id_images_root), 'images', path) for path in id_images]
        for path in id_images:
            assert os.path.isfile(path)
        id_dataset = ListDatasetWithIndex(id_images, flip_color=True)
    else:
        raise ValueError(args.id_images_root)

    if args.save_root is None:
        runname_name = "_".join(args.ckpt_path.split('/')[-4:-2])
        id_name = os.path.basename(args.id_images_root).split('.')[0]
        args.save_root = os.path.join(os.path.dirname(args.id_images_root), 'dataset', runname_name, f'id:{id_name}_sty:{args.style_sampling_method}')
        os.makedirs(args.save_root, exist_ok=True)

    if args.use_writer:
        if args.num_partition > 1:
            args.save_root = os.path.join(args.save_root, f"record_{args.partition_idx}-{args.num_partition}")
        else:
            args.save_root = os.path.join(args.save_root, 'record')
        print('using writer', args.save_root)
        writer = Writer(args.save_root)
    else:
        writer = None

    dataset_generate_mimic_train(pl_module, style_dataset, id_dataset,
                     num_image_per_subject=args.num_image_per_subject, num_subject=args.num_subject,
                     batch_size=args.batch_size, num_workers=args.num_workers, save_root=args.save_root,
                     style_sampling_method=args.style_sampling_method,
                     num_partition=args.num_partition, partition_idx=args.partition_idx, writer=writer,
                     )

    if args.use_writer:
        writer.close()

if __name__ == "__main__":
    main()
