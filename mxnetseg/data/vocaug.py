# coding=utf-8

import os
import numpy as np
import mxnet as mx
from PIL import Image
from gluoncv.data.segbase import SegmentationDataset
from mxnetseg.tools import DATASETS, dataset_dir


@DATASETS.add_component
class PascalVOCAug(SegmentationDataset):
    """
    Fusion dataset of Pascal VOC 2012 and SBD.
    """

    NUM_CLASS = 21

    def __init__(self, root=None, split='train', mode=None, transform=None, **kwargs):
        root = root if root is not None else os.path.join(dataset_dir(), 'VOCdevkit', 'VOCAug')
        super(PascalVOCAug, self).__init__(root, split, mode, transform, **kwargs)
        _img_dir = os.path.join(root, 'img_aug')
        _mask_dir = os.path.join(root, 'cls_aug')
        if split == 'train':
            # _split_f = os.path.join(root, 'trainval_aug.txt')
            _split_f = os.path.join(root, 'train_aug.txt')
        elif split == 'val':
            _split_f = os.path.join(root, 'val.txt')
        else:
            raise RuntimeError('Unknown dataset split: {}'.format(split))

        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), 'r') as lines:
            for line in lines:
                _image_name, _mask_name = line.strip().split(' ')
                _image = os.path.join(_img_dir, _image_name)
                assert os.path.isfile(_image)
                self.images.append(_image)
                _mask = os.path.join(_mask_dir, _mask_name)
                assert os.path.isfile(_mask)
                self.masks.append(_mask)

        assert len(self.images) == len(self.masks)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx])
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalise and ToTensor
        if self.transform is not None:
            img = self.transform(img)

        return img, mask

    def __len__(self):
        return len(self.images)

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return mx.nd.array(target, mx.cpu(0))

    @property
    def classes(self):
        return ('background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                'motorcycle', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
                'tv')
