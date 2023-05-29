import os.path
from typing import Any, Dict, Optional, Tuple
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from src.dataset import face_dataset
from src.dataset import encoded_dataset

class FaceDataModule(LightningDataModule):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return len(self.data_train.label_groups)

    def prepare_data(self):
        image_dataset_path = os.path.join(self.hparams.data_dir, self.hparams.dataset_name)
        if self.hparams.record_file_type == 'encoded':
            encoded_dataset.maybe_make_train_rec(image_dataset_path, self.hparams, self.trainer.model)

    def setup(self, stage: Optional[str] = None):
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset_path = os.path.join(self.hparams.data_dir, self.hparams.dataset_name)
            encoded_rec = encoded_dataset.maybe_load_train_rec(dataset_path, self.hparams)
            self.data_train = face_dataset.make_dataset(dataset_path,
                                                        deterministic=False,
                                                        img_size=self.hparams.img_size,
                                                        return_extra_same_label_samples=self.hparams.return_extra_same_label_samples,
                                                        subset=self.hparams.train_val_split[0],
                                                        orig_augmentations1=self.hparams.orig_augmentations1,
                                                        orig_augmentations2=self.hparams.orig_augmentations2,
                                                        encoded_rec=encoded_rec,
                                                        return_identity_image=self.hparams.return_identity_image,
                                                        return_face_contour=self.hparams.return_face_contour,
                                                        trim_outlier=self.hparams.trim_outlier
                                                        )
            self.data_val = face_dataset.make_dataset(dataset_path,
                                                        deterministic=True,
                                                        img_size=self.hparams.img_size,
                                                        return_extra_same_label_samples=self.hparams.return_extra_same_label_samples,
                                                        subset=self.hparams.train_val_split[1],
                                                        return_identity_image=self.hparams.return_identity_image,
                                                        return_face_contour=False,
                                                        trim_outlier=False)
            self.data_test = self.data_val
            print('train data:', len(self.data_train))
            print('val data:', len(self.data_val))
            print('test_data:', len(self.data_test))


    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "casia_webface.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
