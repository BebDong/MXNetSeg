# coding=utf-8
# Adapted from: https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/pascal_aug/segmentation.py

import os
import scipy.io
from PIL import Image
from gluoncv.data.segbase import SegmentationDataset
from mxnetseg.utils import DATASETS, dataset_dir


@DATASETS.add_component
class SBD(SegmentationDataset):
    """
    B. Hariharan, P. Arbelaez, L. D. Bourdev, S. Maji, and J. Malik.
        Semantic contours from inverse detectors. In ICCV, 2011
    """

    NUM_CLASS = 21

    def __init__(self, root=None, split='train', mode=None, transform=None, **kwargs):
        root = root if root is not None else os.path.join(dataset_dir(), 'VOCdevkit', 'SBD')
        super(SBD, self).__init__(root, split, mode, transform, **kwargs)
        # train/val/test splits are pre-cut
        _mask_dir = os.path.join(root, 'cls')
        _image_dir = os.path.join(root, 'img')
        if split == 'train':
            _split_f = os.path.join(root, 'trainval.txt')
            # _split_f = os.path.join(root, 'train.txt')
        elif split == 'val':
            _split_f = os.path.join(root, 'val.txt')
        else:
            raise RuntimeError('Unknown dataset split: {}'.format(split))

        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in lines:
                _image = os.path.join(_image_dir, line.rstrip('\n') + ".jpg")
                assert os.path.isfile(_image)
                self.images.append(_image)
                _mask = os.path.join(_mask_dir, line.rstrip('\n') + ".mat")
                assert os.path.isfile(_mask)
                self.masks.append(_mask)

        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = self._load_mat(self.masks[index])
        # synchronized transform
        if self.mode == 'train':
            img, target = self._sync_transform(img, target)
        elif self.mode == 'val':
            img, target = self._val_sync_transform(img, target)
        else:
            raise RuntimeError('unknown mode for dataloader: {}'.format(self.mode))
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    @staticmethod
    def _load_mat(filename):
        mat = scipy.io.loadmat(filename, mat_dtype=True, squeeze_me=True,
                               struct_as_record=False)
        mask = mat['GTcls'].Segmentation
        return Image.fromarray(mask)

    def __len__(self):
        return len(self.images)

    @property
    def classes(self):
        """Category names."""
        return ('background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                'motorcycle', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
                'tv')
