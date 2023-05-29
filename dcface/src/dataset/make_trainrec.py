import os
import mxnet as mx
import numpy as np
import torch
try:
    import Queue as queue
except ImportError:
    import queue

FNAME_REC = 'file.rec'
FNAME_IDX = 'file.idx'
DONE_NAME = 'done.pth'

class FeatureSaver():
    def __init__(self, save_root):

        self.save_root = save_root
        self.record = mx.recordio.MXIndexedRecordIO(os.path.join(save_root, FNAME_IDX),
                                                    os.path.join(save_root, FNAME_REC), 'w')
        self.q_out = queue.Queue()
        self.i = 0
        self.encoded_shape = None

    def feature_encode(self, feature):
        assert feature.ndim == 3  # CxHxW
        if self.encoded_shape is None:
            self.encoded_shape = list(feature.shape)
        header = mx.recordio.IRHeader(0, 0, self.i, 0)
        save_features = feature.view(-1)
        save_features_np = save_features.detach().cpu().numpy()
        save_features_np_fp16 = save_features_np.astype(np.float16)
        s = mx.recordio.pack(header, save_features_np_fp16.tobytes())
        self.q_out.put((self.i, s, [self.i, self.i]))
        self.record.write_idx(self.i, s)
        self.i = self.i + 1

    def mark_done(self):
        torch.save({'shape': self.encoded_shape}, os.path.join(self.save_root, DONE_NAME))

class FeatureReader():

    def __init__(self, root='/mckim/temp/temp_recfiles'):
        path_imgidx = os.path.join(root, FNAME_IDX)
        path_imgrec = os.path.join(root, FNAME_REC)
        self.record = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        self.feature_shape = torch.load(os.path.join(root, DONE_NAME))['shape']

    def read_by_index(self, index):
        header, binary = mx.recordio.unpack(self.record.read_idx(index))
        feature = np.frombuffer(binary, dtype=np.float16)
        feature = torch.tensor(feature.astype(np.float32))
        feature = feature.reshape(*self.feature_shape)
        return feature

