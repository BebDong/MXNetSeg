# coding=utf-8

import os
import numpy as np
import mxnet as mx
from PIL import Image
from gluoncv.data.segbase import SegmentationDataset
from mxnetseg.tools import DATASETS, dataset_dir


@DATASETS.add_component
class CamVid(SegmentationDataset):
    """
    G. J. Brostow, et al. Semantic object classes in video:
    A high-definition ground truth database. Pattern Recognit. Lett., 2009.
    """

    NUM_CLASS = 11

    def __init__(self, root=None, split='train', mode=None, transform=None,
                 full_resolution=False, **kwargs):
        base_dir = 'CamVidFull' if full_resolution else 'CamVid'
        root = root if root is not None else os.path.join(dataset_dir(), base_dir)
        super(CamVid, self).__init__(root, split, mode, transform, **kwargs)
        _img_dir = os.path.join(root, 'images')
        _mask_dir = os.path.join(root, 'labels')
        if split == 'train':
            _split_f = os.path.join(root, 'trainval.txt')
        elif split == 'val':
            _split_f = os.path.join(root, 'test.txt')
        elif split == 'test':
            _split_f = os.path.join(root, 'test.txt')
        else:
            raise RuntimeError(f'Unknown dataset split: {split}')

        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), 'r') as lines:
            for line in lines:
                _img_name, _mask_name = line.strip().split(' ')

                _image = os.path.join(_img_dir, _img_name)
                assert os.path.isfile(_image)
                self.images.append(_image)

                _mask = os.path.join(_mask_dir, _mask_name)
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
        target[target == 11] = -1
        return mx.nd.array(target, mx.cpu(0))

    def __len__(self):
        return len(self.images)

    @property
    def classes(self):
        return ('sky', 'building', 'column_pole', 'road', 'sidewalk', 'tree',
                'sign', 'fence', 'car', 'pedestrian', 'byciclist', 'void')
