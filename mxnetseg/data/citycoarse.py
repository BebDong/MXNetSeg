# coding=utf-8

import os
import mxnet as mx
import numpy as np
from PIL import Image
from gluoncv.data.segbase import SegmentationDataset
from mxnetseg.utils import DATASETS, dataset_dir


@DATASETS.add_component
class CityCoarse(SegmentationDataset):
    """
    Cityscapes dataset with coarse labeled data.
    """
    NUM_CLASS = 19

    def __init__(self, root=None, split='train', mode=None, transform=None, **kwargs):
        root = root if root is not None else os.path.join(dataset_dir(), 'Cityscapes')
        super(CityCoarse, self).__init__(root, split, mode, transform, **kwargs)
        assert self.mode in ('train', 'val')
        self.images, self.masks = _get_city_pairs(self.root, self.split)
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index])
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            raise RuntimeError('unknown mode for dataloader: {}'.format(self.mode))
        if self.transform is not None:
            img = self.transform(img)
        return img, mask

    def _mask_transform(self, mask):
        target = _class_to_index(np.array(mask).astype('int32'))
        return mx.nd.array(target, mx.cpu(0))

    @property
    def classes(self):
        return ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                'traffic light', 'traffic sign', 'vegetation', 'terrain',
                'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                'motorcycle', 'bicycle')

    def __len__(self):
        return len(self.images)


def _class_to_index(mask):
    key = np.array([-1, -1, -1, -1, -1, -1,
                    -1, -1, 0, 1, -1, -1,
                    2, 3, 4, -1, -1, -1,
                    5, -1, 6, 7, 8, 9,
                    10, 11, 12, 13, 14, 15,
                    -1, -1, 16, 17, 18])
    mapping = np.array(range(-1, len(key) - 1)).astype('int32')
    values = np.unique(mask)
    for value in values:
        assert (value in mapping)
    index = np.digitize(mask.ravel(), mapping, right=True)
    return key[index].reshape(mask.shape)


def _get_city_pairs(folder, split='train'):
    img_paths = []
    mask_paths = []
    if split == 'train':
        img_folder = os.path.join(folder, 'leftImg8bit/train_extra')
        mask_folder = os.path.join(folder, 'gtCoarse/train_extra')
        name_suffix = 'gtCoarse_labelIds'
    else:
        img_folder = os.path.join(folder, 'leftImg8bit/val')
        mask_folder = os.path.join(folder, 'gtFine/val')
        name_suffix = 'gtFine_labelIds'
    for root, directories, files in os.walk(img_folder):
        for filename in files:
            if filename.endswith(".png"):
                imgpath = os.path.join(root, filename)
                foldername = os.path.basename(os.path.dirname(imgpath))
                maskname = filename.replace('leftImg8bit', name_suffix)
                maskpath = os.path.join(mask_folder, foldername, maskname)
                if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                else:
                    print('cannot find the mask or image:', imgpath, maskpath)
    return img_paths, mask_paths
