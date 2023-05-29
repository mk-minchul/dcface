import os
import pandas as pd
import mxnet as mx
import cv2

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
            cv2.imwrite(img_save_path, image[:,:,::-1])

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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rec_path', type=str, default='dcface_0.5m_oversample_xid/record')
    parser.add_argument('--save_path', type=str, default='dcface_0.5m_oversample_xid/images')
    args = parser.parse_args()

    reader = RecordReader(root=args.rec_path)
    reader.export(args.save_path)
