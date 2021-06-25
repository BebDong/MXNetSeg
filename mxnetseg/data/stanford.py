# coding=utf-8

import os
import numpy as np
import mxnet as mx
from PIL import Image
from gluoncv.data.segbase import SegmentationDataset
from mxnetseg.tools import DATASETS, dataset_dir


@DATASETS.add_component
class StanfordBackground(SegmentationDataset):
    """
    Stanford background dataset.
        B. Liu, et al. Single Image Depth Estimation from Predicted Semantic Labels.
    """

    NUM_CLASS = 8

    def __init__(self, root=None, split='train', mode=None, transform=None, **kwargs):
        root = root if root is not None else os.path.join(dataset_dir(), 'Stanford10')
        super(StanfordBackground, self).__init__(root, split, mode, transform, **kwargs)
        _img_dir = os.path.join(root, 'images')
        _mask_dir = os.path.join(root, 'labels')
        if split == 'train':
            _split_f = os.path.join(root, 'trainList.txt')
        elif split == 'val':
            _split_f = os.path.join(root, 'evalList.txt')
        else:
            raise RuntimeError('Unknown dataset split: {}'.format(split))

        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), 'r') as lines:
            for line in lines:
                _file_name = line.strip()

                _image = os.path.join(_img_dir, _file_name + '.jpg')
                assert os.path.isfile(_image)
                self.images.append(_image)

                _mask = os.path.join(_mask_dir, _file_name + '.png')
                assert os.path.isfile(_mask)
                self.masks.append(_mask)
        assert len(self.images) == len(self.masks)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx])
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[idx])
        # synchronized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        return img, mask

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return mx.nd.array(target, mx.cpu(0))

    def __len__(self):
        return len(self.images)

    @property
    def classes(self):
        return ('Sky', 'tree/bush', 'road/path', 'grass', 'water',
                'building', 'mountain', 'foreground object')
