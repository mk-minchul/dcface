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
# assert os.path.isdir(os.getenv('DATA_ROOT'))

LOG_ROOT = str(root.parent.parent / 'experiments')
os.environ.update({'LOG_ROOT': LOG_ROOT})
os.environ.update({'PROJECT_TASK': root.stem})
os.environ.update({'REPO_ROOT': str(root.parent.parent)})

import hydra
from pytorch_lightning import LightningModule, LightningDataModule
from argparse import ArgumentParser
import omegaconf
from src.visualizations.extra_visualization import dataset_generate
from src.general_utils.os_utils import get_all_files
from src.visualizations.extra_visualization import ListDatasetWithIndex
import numpy as np
from src.visualizations.record import Writer


def main():

    parser = ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='pretrained_models/dcface_3x3.ckpt')
    parser.add_argument('--num_image_per_subject', type=int, default=1)
    parser.add_argument('--num_subject', type=int, default=1)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_partition', type=int, default=1)
    parser.add_argument('--partition_idx', type=int, default=0)

    parser.add_argument('--save_root', type=str, default=None)
    parser.add_argument('--id_images_root', default='sample_images/id_images/sample_57.png')
    parser.add_argument('--style_images_root', type=str, default='sample_images/style_images/woman')

    parser.add_argument('--style_sampling_method', type=str, default='random',
                        choices=['list', 'feature_sim_center:topk_sampling_top1',
                                 'feature_sim_center:top1_sampling_topk', 'list'])
    parser.add_argument('--use_writer', action='store_true')

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.id_images_root = os.path.join(root, args.id_images_root)
    args.style_images_root = os.path.join(root, args.style_images_root)

    ckpt = torch.load(os.path.join(root, args.ckpt_path))
    model_hparam = ckpt['hyper_parameters']
    model_hparam['unet_config']['params']['pretrained_model_path'] = os.path.join(root, model_hparam['unet_config']['params']['pretrained_model_path'])
    model_hparam['recognition']['ckpt_path'] = os.path.join(root, model_hparam['recognition']['ckpt_path'])
    model_hparam['recognition']['center_path'] = os.path.join(root, model_hparam['recognition']['center_path'])
    model_hparam['recognition_eval']['center_path'] = os.path.join(root, model_hparam['recognition_eval']['center_path'])

    model_hparam['_target_'] = 'src.trainer.Trainer'
    model_hparam['_partial_'] = True

    # load pl_module
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
        datamodule: LightningDataModule = hydra.utils.instantiate(dataconfig)
        datamodule.setup()
        style_dataset = datamodule.data_train
        style_dataset.deterministic = True
    else:
        style_images = get_all_files(args.style_images_root, extension_list=['.png', '.jpg'])
        style_dataset = ListDatasetWithIndex(style_images, flip_color=True)

    # load id images
    if os.path.isdir(args.id_images_root):
        id_images = get_all_files(args.id_images_root, extension_list=['.png', '.jpg'])
        np.random.shuffle(id_images)
        if len(id_images) < args.num_subject:
            id_images = id_images * args.num_subject
        id_dataset = ListDatasetWithIndex(id_images, flip_color=True)
    elif os.path.isfile(args.id_images_root):
        id_images = [args.id_images_root] * len(style_dataset)
        args.num_image_per_subject = len(style_dataset)
        id_dataset = ListDatasetWithIndex(id_images, flip_color=True)
    else:
        raise ValueError(args.id_images_root)

    if args.save_root is None:
        runname_name = os.path.basename(args.ckpt_path).split('.')[0]
        id_name = os.path.basename(args.id_images_root).split('.')[0]
        args.save_root = os.path.join(root, 'generated_images', runname_name, f'id:{id_name}_sty:{args.style_sampling_method}')
        os.makedirs(args.save_root, exist_ok=True)

    print('saving at {}'.format(args.save_root))
    if args.use_writer:
        if args.num_partition > 1:
            args.save_root = os.path.join(args.save_root, f"record_{args.partition_idx}-{args.num_partition}")
        else:
            args.save_root = os.path.join(args.save_root, 'record')
        print('using writer', args.save_root)
        writer = Writer(args.save_root)
    else:
        writer = None

    dataset_generate(pl_module, style_dataset, id_dataset,
                     num_image_per_subject=args.num_image_per_subject, num_subject=args.num_subject,
                     batch_size=args.batch_size, num_workers=args.num_workers, save_root=args.save_root,
                     style_sampling_method=args.style_sampling_method,
                     num_partition=args.num_partition, partition_idx=args.partition_idx, writer=writer)

    if args.use_writer:
        writer.close()

if __name__ == "__main__":
    main()
