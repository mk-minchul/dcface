import numbers
import mxnet as mx
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import torch
from PIL import Image
import pandas as pd
import cv2
from src.general_utils.os_utils import get_all_files
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
from PIL import Image
from scipy import misc


def read_list(path_in):
    """Reads the .lst file and generates corresponding iterator.
    Parameters
    ----------
    path_in: string
    Returns
    -------
    item iterator that contains information in .lst file
    returns [idx, label, path]
    """
    with open(path_in) as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            line = [i.strip() for i in line.strip().split('\t')]
            line_len = len(line)
            # check the data format of .lst file
            assert line_len == 3
            item = {'idx': int(line[0]), "path": line[2], 'label': float(line[1])}
            yield item


class BaseMXDataset(Dataset):
    def __init__(self, root_dir, swap_color_order=False, resolution=112):
        super(BaseMXDataset, self).__init__()
        self.to_PIL = transforms.ToPILImage()
        self.root_dir = root_dir
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        path_imglst = os.path.join(root_dir, 'train.lst')

        self.record = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

        # grad image index from the record and know how many images there are.
        # image index could be occasionally random order. like [4,3,1,2,0]
        s = self.record.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.record.keys))
        print('self.imgidx length', len(self.imgidx))

        try:
            success = True
            list(read_list(path_imglst))
        except:
            success = False
        if os.path.isfile(path_imglst) and success:
            self.record_info = pd.DataFrame(list(read_list(path_imglst)))
            self.insightface_trainrec = False
        else:
            self.insightface_trainrec = True
            # make one yourself
            record_info = []
            for idx in self.imgidx:
                s = self.record.read_idx(idx)
                header, _ = mx.recordio.unpack(s)
                label = header.label
                row = {'idx': idx, 'path': '{}/name.jpg'.format(label), 'label': label}
                record_info.append(row)
            self.record_info = pd.DataFrame(record_info)

        self.swap_color_order = swap_color_order
        if self.swap_color_order:
            print('[INFO] Train data in swap_color_order')
        self.resolution = resolution
        print('[INFO] input image resolution : {}'.format(resolution))

    def read_sample(self, index):
        idx = self.imgidx[index]
        s = self.record.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if sample.shape[0] != self.resolution:
            sample = cv2.resize(sample, (self.resolution, self.resolution))

        if self.swap_color_order:
            # swap rgb to bgr since image is in rgb for webface
            sample = Image.fromarray(np.asarray(sample)[:, :, ::-1])
        else:
            assert not 'webface4m' in self.root_dir.lower()
            sample = self.to_PIL(sample)

        return sample, label

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        # return len(self.imgidx)
        raise NotImplementedError()


class LabelConvertedMXFaceDataset(BaseMXDataset):

    def __init__(self,
                 root_dir,
                 swap_color_order=False,
                 rec_label_to_another_label=None,
                 resolution=112
                 ):
        # rec_label_to_another_label: dictionary converting record label to another label like torch ImageFolderLabel
        super(LabelConvertedMXFaceDataset, self).__init__(root_dir=root_dir,
                                                          swap_color_order=swap_color_order,
                                                          resolution=resolution)
        if rec_label_to_another_label is None:
            # make one using path
            # image folder with 0/1.jpg

            # from record file label to folder name
            rec_label = self.record_info.label.tolist()
            foldernames = self.record_info.path.apply(lambda x: x.split('/')[0]).tolist()
            self.rec_to_folder = {}
            for i, j in zip(rec_label, foldernames):
                self.rec_to_folder[i] = j

            # from folder name to number as torch imagefolder
            foldernames = sorted(str(entry) for entry in self.rec_to_folder.values())
            self.folder_to_num = {cls_name: i for i, cls_name in enumerate(foldernames)}
            self.rec_label_to_another_label = {}

            # combine all
            for x in rec_label:
                self.rec_label_to_another_label[x] = self.folder_to_num[self.rec_to_folder[x]]

        else:
            self.rec_label_to_another_label = rec_label_to_another_label

    def __len__(self):
        return len(self.imgidx)

    def read_sample(self, index):
        sample, record_label = super().read_sample(index)
        new_label = self.rec_label_to_another_label[record_label.item()]
        new_label = torch.tensor(new_label, dtype=torch.long)
        return sample, new_label, record_label


class FaceMXDataset(LabelConvertedMXFaceDataset):
    def __init__(self,
                 root_dir,
                 swap_color_order=False,
                 rec_label_to_another_label=None,
                 transform=None,
                 target_transform=None,
                 resolution=112,
                 return_label=False,
                 return_extra_same_label_samples=False,
                 subset='0-all',
                 deterministic=False,
                 encoded_rec=None,
                 return_identity_image='',
                 return_face_contour='',
                 trim_outlier=False
                 ):
        super(FaceMXDataset, self).__init__(root_dir,
                                            swap_color_order=swap_color_order,
                                            rec_label_to_another_label=rec_label_to_another_label,
                                            resolution=resolution)
        if isinstance(transform, list):
            # split transform for returning both 112x112 and 128x128
            transform_random1, transform_random2, transform_determ1, transform_determ2 = transform
            self.transform_random1 = transform_random1
            self.transform_random2 = transform_random2
            self.transform_determ1 = transform_determ1
            self.transform_determ2 = transform_determ2
            self.split_transform = True
        else:
            self.transform = transform
            self.split_transform = False
        self.target_transform = target_transform
        assert self.target_transform is None  # not implemented yet
        self.return_label = return_label
        self.return_extra_same_label_samples = return_extra_same_label_samples

        if return_extra_same_label_samples:
            groupby = self.record_info.groupby('label')
            self.label_groups = {}
            for k, v in groupby.groups.items():
                self.label_groups[k] = np.array(v)

        if subset == '0-all':
            start_index = 0
            end_index = len(self)
        else:
            assert len(subset.split('-')) == 2
            start_index = int(float(subset.split('-')[0]) * len(self))
            end_index = int(float(subset.split('-')[1]) * len(self))
        assert start_index >= 0
        assert end_index <= len(self)
        assert start_index < end_index
        self.start_index = start_index
        self.end_index = end_index

        self.deterministic = deterministic

        self.encoded_rec = encoded_rec

        self.return_identity_image = return_identity_image
        if return_identity_image:
            self.id_image_df = torch.load(os.path.join(os.environ.get('REPO_ROOT'), return_identity_image))['similarity_df']
        else:
            self.id_image_df = None

        self.return_face_contour = return_face_contour
        if return_face_contour:
            print('loading mask')
            self.mask = []
            for i in range(10):
                mask = torch.load(os.path.join(os.environ.get('REPO_ROOT'), return_face_contour, f'mask_{i}.pth'))
                self.mask.append(mask)
            self.mask = np.concatenate(self.mask, axis=0)
            assert len(self.mask) == len(self.imgidx)
            print('done loading mask')
        else:
            self.mask = None

        self.trim_outlier = trim_outlier


    def __len__(self):
        if hasattr(self, 'end_index') and hasattr(self, 'start_index'):
            return self.end_index - self.start_index
        else:
            return len(self.imgidx)

    def transform_images(self, sample):
        if self.split_transform:
            if self.deterministic:
                pass
            else:
                sample = self.transform_random1(sample)
            sample1 = self.transform_determ1(sample)

            # sample2 is usually original shape
            if self.deterministic:
                sample2 = sample
            else:
                sample2 = self.transform_random2(sample)
            sample2 = self.transform_determ2(sample2)

        else:
            raise ValueError('not implemented')

        return sample1, sample2


    def __getitem__(self, index, ):

        index = index + self.start_index
        return_dict = {}

        sample, target, record_label = self.read_sample(index)
        if self.trim_outlier:
            target_df = self.id_image_df[target.item()]
            cossim_to_center = target_df[target_df['data_index'] == index].cossim
            outlier_threshold = target_df.cossim.quantile(0.1)
            is_inlier = cossim_to_center > outlier_threshold
            is_inlier = is_inlier.sample(1).item()
            if not is_inlier:
                # resample
                index = target_df[(target_df['cossim'] > outlier_threshold) &
                                  (target_df['data_index'] > self.start_index) &
                                  (target_df['data_index'] < (self.start_index + len(self)))].sample().data_index.item()
                sample, __target, record_label = self.read_sample(index)
                assert __target == target

        sample, orig_sample = self.transform_images(sample)
        return_dict['image'] = sample
        return_dict['index'] = index
        return_dict['orig'] = orig_sample

        if self.encoded_rec is not None:
            encoded = self.encoded_rec.read_by_index(index)
            return_dict['encoded'] = encoded

        if self.return_label:
            return_dict['class_label'] = target
            return_dict['human_label'] = 'subject_' + str(target.item())


        if self.return_identity_image:
            repeat = 0
            while True:
                good_image_index = self.id_image_df[target.item()].loc[0:3].sample(1)['data_index'].item()
                repeat += 1
                if repeat > 10: print('repeat error')
                if good_image_index != index or repeat > 10:
                    break
            id_image, id_target, id_record_label = self.read_sample(good_image_index)
            id_image, orig_id_image = self.transform_images(id_image)
            assert id_target == target
            return_dict['id_image'] = id_image

        if self.return_face_contour:
            return_dict['face_contour'] = self.mask[index]

        if self.return_extra_same_label_samples:
            same_label_index = self.label_groups[record_label.item()]
            extra_index = np.random.choice(same_label_index, 1)[0]
            extra_sample, extra_target, _ = self.read_sample(extra_index)
            extra_sample, extra_orig_sample = self.transform_images(extra_sample)
            assert extra_target.item() == target.item()
            return_dict['extra_image'] = extra_sample
            return_dict['extra_index'] = extra_index
            return_dict['extra_orig'] = extra_orig_sample
            if self.encoded_rec is not None:
                extra_encoded = self.encoded_rec.read_by_index(extra_index)
                return_dict['extra_encoded'] = extra_encoded

        return return_dict



class ListDatasetWithIndex(Dataset):
    def __init__(self, img_list, subset, transform, target_transform, return_label, deterministic, image_is_saved_with_swapped_B_and_R=False):
        super(ListDatasetWithIndex, self).__init__()

        if subset == '0-all':
            start_index = 0
            end_index = len(img_list)
        else:
            assert len(subset.split('-')) == 2
            start_index = int(float(subset.split('-')[0]) * len(img_list))
            end_index = int(float(subset.split('-')[1]) * len(img_list))
        assert start_index >= 0
        assert end_index <= len(img_list)
        assert start_index < end_index
        self.start_index = start_index
        self.end_index = end_index

        self.img_list = img_list
        self.dummy_labels = np.arange(len(img_list)) % 100
        rows = []
        for idx, (name, label) in enumerate(zip(self.img_list, self.dummy_labels)):
            row = {'idx': idx, 'path': '{}/name.jpg'.format(label), 'label': label}
            rows.append(row)
        self.record_info = pd.DataFrame(rows)

        self.transform = transforms
        self.image_is_saved_with_swapped_B_and_R = image_is_saved_with_swapped_B_and_R

        if isinstance(transform, list):
            # split transform for returning both 112x112 and 128x128
            transform_random1, transform_random2, transform_determ1, transform_determ2 = transform
            self.transform_random1 = transform_random1
            self.transform_random2 = transform_random2
            self.transform_determ1 = transform_determ1
            self.transform_determ2 = transform_determ2
            self.split_transform = True
        else:
            self.transform = transform
            self.split_transform = False
        self.target_transform = target_transform
        assert self.target_transform is None  # not implemented yet
        self.return_label = return_label
        self.deterministic = deterministic

        self.rec_label_to_another_label = dict(zip(self.dummy_labels, self.dummy_labels))

    def __len__(self):
        if hasattr(self, 'end_index') and hasattr(self, 'start_index'):
            return self.end_index - self.start_index
        else:
            return len(self.img_list)

    def transform_images(self, sample):
        if self.split_transform:
            if self.deterministic:
                pass
            else:
                sample = self.transform_random1(sample)
            sample1 = self.transform_determ1(sample)

            # sample2 is usually original shape
            if self.deterministic:
                sample2 = sample
            else:
                sample2 = self.transform_random2(sample)
            sample2 = self.transform_determ2(sample2)

        else:
            raise ValueError('not implemented')

        return sample1, sample2

    def read_image(self, idx):

        if self.image_is_saved_with_swapped_B_and_R:
            with open(self.img_list[idx], 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
        else:
            img = cv2.imread(self.img_list[idx])
            if img is None:
                print(self.img_list[idx])
                raise ValueError(self.img_list[idx])
            img = img[:,:,:3]
            img = Image.fromarray(img)
        return img

    def __getitem__(self, index):
        index = index + self.start_index
        sample = self.read_image(index)
        sample, orig_sample = self.transform_images(sample)
        return_dict = {}
        return_dict['image'] = sample
        return_dict['index'] = index
        return_dict['orig'] = orig_sample

        if self.return_label:
            return_dict['class_label'] = torch.tensor(0)
            return_dict['human_label'] = 'subject_0'

        return return_dict


def make_dataset(data_path,
                 deterministic=False,
                 img_size=112,
                 return_extra_same_label_samples=False,
                 subset='0-all',
                 orig_augmentations1=[],
                 orig_augmentations2=[],
                 encoded_rec=None,
                 return_identity_image='',
                 return_face_contour='',
                 trim_outlier=False,
                 ):


    transform_random1 = []
    if not deterministic and orig_augmentations1:
        for aug in orig_augmentations1:
            prob = float(aug.split(":")[-1])
            if 'flip' in aug:
                t = transforms.RandomApply(transforms=[transforms.RandomHorizontalFlip()], p=prob)
            elif 'gray' in aug:
                t = transforms.RandomApply(transforms=[transforms.Grayscale(num_output_channels=3)], p=prob)
            elif 'photo' in aug:
                t = transforms.RandomApply(transforms=[transforms.ColorJitter(brightness=.3, contrast=.3)], p=prob)
            else:
                raise ValueError('not correct')
            transform_random1.append(t)
    transform_random1 = transforms.Compose(transform_random1)

    transform_random2 = []
    if not deterministic and orig_augmentations2:
        for aug in orig_augmentations2:
            prob = float(aug.split(":")[-1])
            if 'flip' in aug:
                t = transforms.RandomHorizontalFlip()
            elif 'gray' in aug:
                t = transforms.RandomApply(transforms=[transforms.Grayscale(num_output_channels=3)], p=prob)
            elif 'photo' in aug:
                t = transforms.RandomApply(transforms=[transforms.ColorJitter(brightness=.3, contrast=.3)], p=prob)
            else:
                raise ValueError('not correct')
            transform_random2.append(t)
    transform_random2 = transforms.Compose(transform_random2)

    transform_determ1 = [transforms.Resize(img_size),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform_determ1 = transforms.Compose(transform_determ1)
    transform_determ2 = [transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform_determ2 = transforms.Compose(transform_determ2)
    if 'casia' in data_path or 'faces_webface_112x112' in data_path:
        return FaceMXDataset(data_path,
                             swap_color_order=False,
                             rec_label_to_another_label=None,
                             transform=[transform_random1, transform_random2, transform_determ1, transform_determ2],
                             target_transform=None,
                             resolution=112,
                             return_label=True,
                             return_extra_same_label_samples=return_extra_same_label_samples,
                             subset=subset,
                             deterministic=deterministic,
                             encoded_rec=encoded_rec,
                             return_identity_image=return_identity_image,
                             return_face_contour=return_face_contour,
                             trim_outlier=trim_outlier,
                             )
    elif 'ffhq' in data_path:
        assert os.path.isdir(data_path)
        all_files = get_all_files(data_path, extension_list=['.png', '.jpg', '.jpeg'], sorted=True)
        dataset = ListDatasetWithIndex(all_files,
                                       subset=subset,
                                       transform=[transform_random1, transform_random2, transform_determ1, transform_determ2],
                                       target_transform=None,
                                       return_label=True,
                                       deterministic=deterministic,
                                       image_is_saved_with_swapped_B_and_R=True)
        return dataset


