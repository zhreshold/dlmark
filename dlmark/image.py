from __future__ import absolute_import
import os
import numpy as np
import mxnet as mx
try:
    import cPickle as pickle
except ImportError:
    import pickle
# from . import data #import DownloadMultiPartDataset
from .data import DownloadMultiPartDataset
from .utils import get_cpu_count

def preprocess_imagenet_val(image_dir, label_fname, output_dir, image_shape):
    """Save raw Imagenet validation images into numpy format
    """
    import mxnet as mx



class ILSVRC12Val(object):
    def __init__(self, batch_size, repo_dir='',
                 root='~/.dlmark/datasets/ilsvrc12_val'):
        self.dataset = DownloadMultiPartDataset(repo_dir, root)
        self.batch_size = batch_size
        self.num_examples = 50000
        self.num_examples_per_part = 1000
        self.curr_part = -1
        self.X = None
        self.y = None
        assert batch_size < self.num_examples_per_part

    def _load_part(self, part):
        fname = self.dataset[part]
        data = np.load(fname)
        self.X = data['X']
        self.y = data['y']
        self.curr_part = part

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError
        part = idx*self.batch_size//self.num_examples_per_part
        if part != self.curr_part:
            self._load_part(part)
        offset = idx * self.batch_size - part * self.num_examples_per_part
        X, y = (self.X[offset:offset+self.batch_size],
                self.y[offset:offset+self.batch_size])

        if X.shape[0] < self.batch_size:
            self._load_part(part+1)
            n = self.batch_size - X.shape[0]
            X = np.concatenate((X, self.X[:n]), axis=0)
            y = np.concatenate((y, self.y[:n]), axis=0)
        return (X, y)

    def __len__(self):
        return self.num_examples // self.batch_size


class COCOVal2017(object):
    def __init__(self, batch_size, transform, prefix, tmp_dir='~/.dlmark/datasets/cocoval2017'):
        tmp_dir = os.path.abspath(tmp_dir)
        if not os.path.isdir(tmp_dir):
            os.makedirs(tmp_dir)
        tmp_file = os.path.join(tmp_dir, 'cocoval2017_' + prefix + '.npy')
        if not os.path.isfile(tmp_file):
            # create cache to speed up load time
            import gluoncv as gcv
            dataset = gcv.data.COCODetection(splits=('instances_val2017',), skip_empty=False)
            print('Creating cache validation data: ' + tmp_file)
            dataloader = mx.gluon.data.DataLoader(dataset.transform(transform), 1, shuffle=False,
                last_batch="keep", num_workers=get_cpu_count())
            buffer = {}
            idx = 0
            for batch in dataloader:
                buffer[idx] = [x.asnumpy() for x in batch]
                idx += 1
            assert idx == len(dataset)
            print('Cache of validation data saved: ' + tmp_file)
            with open(tmp_file, 'wb') as fid:
                pickle.dump(buffer, fid)
            self._buffer = buffer
        else:
            with open(tmp_file, 'rb') as fid:
                self._buffer = pickle.load(fid)

        assert len(self._buffer.keys()) == 5000
        self.batch_size = batch_size

    def __len__(self):
        return 5000 // self.batch_size

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError
        begin = idx * self.batch_size
        batch = self._buffer[begin]
        if self.batch_size < 2:
            return tuple([mx.nd.NDArray(x) for x in batch])

        N = len(batch)
        batches = [batch]
        for i in range(begin + 1, begin + self.batch_size):
            batches.append(self._buffer[i])

        out = tuple([mx.nd.NDArray(np.concatenate([batches[j] for j in range(N)]))])
        return out


if __name__ == '__main__':
    data = ILSVRC12Val(128, 'http://xx/', root='/home/ubuntu/imagenet_val/')
    print(data[0][0].shape)
    print('xxx')
