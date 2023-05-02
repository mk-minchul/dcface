import os
import mxnet as mx
try:
    import Queue as queue
except ImportError:
    import queue
import numpy as np
import os
import mxnet as mx
import pandas as pd
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class RecordReader():

    def __init__(self, root='/mckim/temp/temp_recfiles'):
        path_imgidx = os.path.join(root, 'file.idx')
        path_imgrec = os.path.join(root, 'file.rec')
        self.root = root
        self.record = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

        path_list = os.path.join(root, 'list.txt')
        info = pd.read_csv(path_list, sep='\t', index_col=0, header=None)
        self.index_to_path = dict(info[1])
        self.path_to_index = {v:k for k,v in self.index_to_path.items()}

    def read_by_index(self, index):
        header, binary = mx.recordio.unpack(self.record.read_idx(index))
        image = mx.image.imdecode(binary).asnumpy()
        path = self.index_to_path[index]
        return image, path

    def read_by_path(self, path):
        index = self.path_to_index[path]
        return self.read_by_index(index)

    def export(self, save_root):
        for idx in self.index_to_path.keys():
            image, path = self.read_by_index(idx)
            img_save_path = os.path.join(save_root, path)
            os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
            cv2.imwrite(img_save_path, image)

    def existing_keys(self):
        return self.path_to_index.keys()

    def load_done_list(self):
        donelist_path = os.path.join(self.root, 'done_list.txt')
        if os.path.isfile(donelist_path):
            donelist = pd.read_csv(donelist_path, header=None, sep='\t')
            donelist.columns = ['type', 'path']
            return set(donelist['path'].values)
        else:
            return None

class SplittedRecordReader():
    def __init__(self, roots):
        print(f'Loading {len(roots)} records')
        print(roots)
        self.records = [RecordReader(root) for root in roots]
        self.path_to_record_num = {}
        for record_idx, record in enumerate(self.records):
            for key in record.path_to_index.keys():
                self.path_to_record_num[key] = record_idx

    def read_by_index(self, index):
        raise NotImplementedError('')

    def read_by_path(self, path):
        record_num = self.path_to_record_num[path]
        return self.records[record_num].read_by_path(path)

    def export(self, save_root):
        raise NotImplementedError('')

    def existing_keys(self):
        return self.path_to_record_num.keys()

    def load_done_list(self):
        donelist = set()
        for record in self.records:
            _donelist = record.load_done_list()
            if _donelist is not None:
                donelist = donelist | _donelist
        return donelist



class RecordDatasetWithIndex(Dataset):
    def __init__(self, img_list, record_dataset):
        super(RecordDatasetWithIndex, self).__init__()
        self.img_list = img_list
        self.record_dataset = record_dataset

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img, path = self.record_dataset.read_by_path(self.img_list[idx])
        if img is None:
            print(self.img_list[idx])
            raise ValueError(self.img_list[idx])
        img = img[:,:,:3]
        img = Image.fromarray(img)
        img = self.transform(img)
        # from utils import img_utils
        # cv2.imwrite('/mckim/temp/temp.png', img_utils.tensor_to_numpy(img))
        return img, idx

def prepare_record_dataloader(img_list, record_dataset, batch_size, num_workers=0):
    image_dataset = RecordDatasetWithIndex(img_list, record_dataset)
    dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    return dataloader


def prepare_record_saver(save_root):

    os.makedirs(save_root, exist_ok=True)
    q_out = queue.Queue()
    fname_rec = 'file.rec'
    fname_idx = 'file.idx'
    fname_list = 'list.txt'
    done_list = 'done_list.txt'
    if os.path.isfile(os.path.join(save_root, fname_idx)):
        os.remove(os.path.join(save_root, fname_idx))
    if os.path.isfile(os.path.join(save_root, fname_rec)):
        os.remove(os.path.join(save_root, fname_rec))
    if os.path.isfile(os.path.join(save_root, fname_list)):
        os.remove(os.path.join(save_root, fname_list))
    if os.path.isfile(os.path.join(save_root, done_list)):
        os.remove(os.path.join(save_root, done_list))

    record = mx.recordio.MXIndexedRecordIO(os.path.join(save_root, fname_idx),
                                           os.path.join(save_root, fname_rec), 'w')
    list_writer = open(os.path.join(save_root, fname_list), 'w')
    mark_done_writer = open(os.path.join(save_root, done_list), 'w')

    return record, q_out, list_writer, mark_done_writer


class Writer():

    def __init__(self, save_root):
        record, q_out, list_writer, mark_done_writer = prepare_record_saver(save_root)
        self.record = record
        self.list_writer = list_writer
        self.mark_done_writer = mark_done_writer  # needed for continuing
        self.q_out = q_out
        self.image_index = 0


    def write(self, rgb_pil_img, save_path):
        header = mx.recordio.IRHeader(0, 0, self.image_index, 0)
        s = mx.recordio.pack_img(header, np.array(rgb_pil_img), quality=100, img_fmt='.jpg')
        item = [self.image_index, save_path]
        self.q_out.put((item[0], s, item))
        _, s, _ = self.q_out.get()
        self.record.write_idx(item[0], s)
        line = '%d\t' % item[0] + '%s\n' % item[1]
        self.list_writer.write(line)
        self.image_index = self.image_index + 1


    def close(self):
        self.record.close()
        self.list_writer.close()
        self.mark_done_writer.close()


    def mark_done(self, context, name):
        line = '%s\t' % context + '%s\n' % name
        self.mark_done_writer.write(line)

