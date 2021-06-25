# coding=utf-8

import os
import numpy as np
import mxnet as mx
from PIL import Image
from gluoncv.data.segbase import SegmentationDataset
from mxnetseg.tools import DATASETS, dataset_dir


@DATASETS.add_component
class Mapillary(SegmentationDataset):
    """
    Mapillary Vistas Dataset for segmentatioin.
    We do not take in to account the unlabelled class when training.
    66-class mIoU = 65-class mIoU * 65 / 66, which is similar to Pascal Context.
    Reference: G. Neuhold, T. Ollmann, S. R. Bulo, and P. Kontschieder, “The Mapillary
        Vistas Dataset for Semantic Understanding of Street Scenes,” in IEEE International
        Conference on Computer Vision, 2017, pp. 5000–5009.
    """
    NUM_CLASS = 65

    def __init__(self, root=None, split='train', mode=None, transform=None, **kwargs):
        root = root if root is not None else os.path.join(dataset_dir(), 'Mapillary')
        super(Mapillary, self).__init__(root, split, mode, transform, **kwargs)
        self.images, self.masks = _get_mapi_pairs(self.root, self.split)

    def __getitem__(self, item):
        img = Image.open(self.images[item]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[item])
        mask = Image.open(self.masks[item])
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
        target[target == 65] = -1  # ignore unlabeled
        return mx.nd.array(target, mx.cpu(0))

    @property
    def classes(self):
        return ('Bird', 'Ground Animal', 'Curb', 'Fence', 'Guard Rail', 'Barrier', 'Wall',
                'Bike Lane', 'Crosswalk-Plain', 'Curb Cut', 'Parking', 'Pedestrian Area',
                'Rail Track', 'Road', 'Service Lane', 'Sidewalk', 'Bridge', 'Building', 'Tunnel',
                'Person', 'Bicyclist', 'Motorcyclist', 'Other Rider', 'Lane Marking-Crosswalk',
                'Lane Marking-General', 'Mountain', 'Sand', 'Sky', 'Snow', 'Terrain', 'Vegetation',
                'Water', 'Banner', 'Bench', 'Bike Rack', 'Billboard', 'Catch Basin', 'CCTV Camera',
                'Fire Hydrant', 'Junction Box', 'Mailbox', 'Manhole', 'Phone Booth', 'Pothole',
                'Street Light', 'Pole', 'Traffic Sign Frame', 'Utility Pole', 'Traffic Light',
                'Traffic Sign (Back)', 'Traffic Sign (Front)', 'Trash Can', 'Bicycle', 'Boat',
                'Bus', 'Car', 'Caravan', 'Motorcycle', 'On Rails', 'Other Vehicle', 'Trailer',
                'Truck', 'Wheeled Slow', 'Car Mount', 'Ego Vehicle')

    def __len__(self):
        return len(self.images)


def _get_mapi_pairs(root, split='train'):
    """
    get all images and masks paths.
    """
    if split in ('train', 'val', 'test'):
        img_folder = os.path.join(root, split, 'images')
        mask_folder = os.path.join(root, split, 'labels')
        img_paths, mask_paths = _get_path_pairs(img_folder, mask_folder)
        return img_paths, mask_paths
    elif split == 'trainval':
        train_img_folder = os.path.join(root, 'train', 'images')
        train_mask_folder = os.path.join(root, 'train', 'labels')
        train_img_paths, train_mask_paths = _get_path_pairs(train_img_folder, train_mask_folder)
        val_img_folder = os.path.join(root, 'val', 'images')
        val_mask_folder = os.path.join(root, 'val', 'labels')
        val_img_paths, val_mask_paths = _get_path_pairs(val_img_folder, val_mask_folder)
        img_paths = train_img_paths + val_img_paths
        mask_paths = train_mask_paths + val_mask_paths
        return img_paths, mask_paths
    else:
        raise RuntimeError(f"Unknown split: {split}")


def _get_path_pairs(img_folder, mask_folder):
    """
    get image/mask path pairs.
    note that labels of 'test' split is empty.
    """
    img_paths = []
    mask_paths = []
    for root, _, files in os.walk(img_folder):
        for filename in files:
            if filename.endswith('.jpg'):
                img_path = os.path.join(root, filename)
                file_id = filename.split('.')[0].strip()
                mask_path = os.path.join(mask_folder, file_id + '.png')
                if os.path.isfile(img_path):
                    img_paths.append(img_path)
                if os.path.isfile(mask_path):
                    mask_paths.append(mask_path)
    return img_paths, mask_paths
