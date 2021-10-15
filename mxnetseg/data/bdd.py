# coding=utf-8

import os
import numpy as np
import mxnet as mx
from PIL import Image
from gluoncv.data.segbase import SegmentationDataset
from mxnetseg.utils import DATASETS, dataset_dir


@DATASETS.add_component
class BDD100K(SegmentationDataset):
    """
    BDD100k for semantic segmentation
    Reference: F. Yu et al., “BDD100K: A Diverse Driving Dataset for
        Heterogeneous Multitask Learning,” in IEEE Conference on Computer
        Vision and Pattern Recognition, 2020, pp. 2633–2642.
    """

    NUM_CLASS = 19

    def __init__(self, root=None, split='train', mode=None, transform=None, **kwargs):
        root = root if root is not None else os.path.join(dataset_dir(), 'BDD', 'seg')
        super(BDD100K, self).__init__(root, split, mode, transform, **kwargs)
        self.images, self.masks = _get_bdd_pairs(root, self.split)
        if split in ('train', 'val'):
            assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])
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

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return mx.nd.array(target, mx.cpu(0))

    def __len__(self):
        return len(self.images)

    @property
    def classes(self):
        return ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                'traffic light', 'traffic sign', 'vegetation', 'terrain',
                'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                'motorcycle', 'bicycle')


def _get_bdd_pairs(folder, split='train'):
    img_folder = os.path.join(folder, 'images', split)
    mask_folder = os.path.join(folder, 'labels', split)
    img_paths, mask_paths = _get_path_pairs(img_folder, mask_folder, split=split)
    return img_paths, mask_paths


def _get_path_pairs(img_folder, mask_folder, split):
    img_paths = []
    mask_paths = []
    for root, _, files in os.walk(img_folder):
        for filename in files:
            img_path = os.path.join(root, filename)
            if os.path.isfile(img_path):
                img_paths.append(img_path)
            else:
                raise RuntimeError(f"Unable to find image: {img_path}")
            if split != 'test':
                mask_name = filename.replace('.jpg', '_train_id.png')
                mask_path = os.path.join(mask_folder, mask_name)
                if os.path.isfile(mask_path):
                    mask_paths.append(mask_path)
                else:
                    raise RuntimeError(f"Unable to find mask: {mask_path}")
    return img_paths, mask_paths
