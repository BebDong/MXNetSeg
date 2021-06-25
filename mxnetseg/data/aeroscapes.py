# coding=utf-8

import os
from PIL import Image
from gluoncv.data.segbase import SegmentationDataset
from mxnetseg.tools import DATASETS, dataset_dir


@DATASETS.add_component
class Aeroscapes(SegmentationDataset):
    """
    Reference:
        Nigam, Ishan, Chen Huang, and Deva Ramanan. "Ensemble knowledge transfer
        for semantic segmentation." 2018 IEEE Winter Conference on Applications
        of Computer Vision (WACV). IEEE, 2018.
    """

    NUM_CLASS = 12

    def __init__(self, root=None, split='train', mode=None, transform=None, **kwargs):
        root = root if root is not None else os.path.join(dataset_dir(), 'Aeroscapes')
        super(Aeroscapes, self).__init__(root, split, mode, transform, **kwargs)
        _img_dir = os.path.join(root, 'JPEGImages')
        _mask_dir = os.path.join(root, 'SegmentationClass')
        _splits_dir = os.path.join(root, 'ImageSets')
        if split == 'train':
            _split_f = os.path.join(_splits_dir, 'trn.txt')
        elif split == 'val':
            _split_f = os.path.join(_splits_dir, 'val.txt')
        else:
            raise RuntimeError('Unknown dataset split.')

        self.images = []
        self.masks = []
        with open(os.path.join(_split_f, ), 'r') as lines:
            for line in lines:
                _image = os.path.join(_img_dir, line.rstrip('\n') + '.jpg')
                assert os.path.isfile(_image)
                self.images.append(_image)
                _mask = os.path.join(_mask_dir, line.rstrip('\n') + ".png")
                assert os.path.isfile(_mask)
                self.masks.append(_mask)
        assert len(self.images) == len(self.masks)

    def __getitem__(self, item):
        img = Image.open(self.images[item]).convert('RGB')
        mask = Image.open(self.masks[item])
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[item])
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)

        if self.transform is not None:
            img = self.transform(img)

        return img, mask

    def __len__(self):
        return len(self.images)

    @property
    def classes(self):
        return ('background', 'person', 'bike', 'car', 'drone', 'boat', 'animal',
                'obstacle', 'construction', 'vegetation', 'road', 'sky')
