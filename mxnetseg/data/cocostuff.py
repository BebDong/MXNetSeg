# coding=utf-8

import os
import scipy.io
import numpy as np
import mxnet as mx
from PIL import Image
from gluoncv.data.segbase import SegmentationDataset
from mxnetseg.utils import DATASETS, dataset_dir


@DATASETS.add_component
class COCOStuff(SegmentationDataset):
    NUM_CLASS = 182

    def __init__(self, root=None, split='train', mode=None, transform=None, **kwargs):
        root = root if root is not None else os.path.join(dataset_dir(), 'COCOStuff')
        super(COCOStuff, self).__init__(root, split, mode, transform, **kwargs)
        _image_dir = os.path.join(root, 'images')
        _mask_dir = os.path.join(root, 'annotations')
        if split == 'train':
            _split_f = os.path.join(root, 'imageLists', 'train.txt')
        elif split == 'val':
            _split_f = os.path.join(root, 'imageLists', 'test.txt')
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

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        target = self._load_mat(self.masks[idx])
        # synchronized transform
        if self.mode == 'train':
            img, target = self._sync_transform(img, target)
        elif self.mode == 'val':
            img, target = self._val_sync_transform(img, target)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(target)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32') - 1
        return mx.nd.array(target, mx.cpu(0))

    @staticmethod
    def _load_mat(filename):
        mat = scipy.io.loadmat(filename, mat_dtype=True, squeeze_me=True,
                               struct_as_record=False)
        mask = mat['S']
        return Image.fromarray(mask)

    def __len__(self):
        return len(self.images)

    @property
    def classes(self):
        return ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter',
                'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate',
                'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door',
                'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush', 'hair brush', 'banner', 'blanket', 'branch',
                'bridge', 'building-other', 'bush', 'cabinet', 'cage', 'cardboard', 'carpet',
                'ceiling-other', 'ceiling-tile', 'cloth', 'clothes', 'clouds', 'counter', 'cupboard',
                'curtain', 'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble', 'floor-other',
                'floor-stone', 'floor-tile', 'floor-wood', 'flower', 'fog', 'food-other', 'fruit',
                'furniture-other', 'grass', 'gravel', 'ground-other', 'hill', 'house', 'leaves',
                'light', 'mat', 'metal', 'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net',
                'paper', 'pavement', 'pillow', 'plant-other', 'plastic', 'platform', 'playingfield',
                'railing', 'railroad', 'river', 'road', 'rock', 'roof', 'rug', 'salad', 'sand', 'sea',
                'shelf', 'sky-other', 'skyscraper', 'snow', 'solid-other', 'stairs', 'stone', 'straw',
                'structural-other', 'table', 'tent', 'textile-other', 'towel', 'tree', 'vegetable',
                'wall-brick', 'wall-concrete', 'wall-other', 'wall-panel', 'wall-stone', 'wall-tile',
                'wall-wood', 'water-other', 'waterdrops', 'window-blind', 'window-other', 'wood')
