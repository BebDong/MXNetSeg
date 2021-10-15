# coding=utf-8

import numpy as np
from tqdm import tqdm

from .ade20k import ADE20K
from .aeroscapes import Aeroscapes
from .bdd import BDD100K
from .camvid import CamVid
from .citycoarse import CityCoarse
from .cityscapes import Cityscapes
from .cocostuff import COCOStuff
from .gatech import GATECH
from .mapillary import Mapillary
from .mhp import MHPV1
from .mscoco import MSCOCO
from .nyuv2 import NYUv2
from .pcontext import PascalContext
from .sbd import SBD
from .siftflow import SiftFlow
from .stanford import StanfordBackground
from .sunrgbd import SUNRGBD
from .voc import PascalVOC
from .vocaug import PascalVOCAug
from .weizhorses import WeizmannHorses

from mxnetseg.utils import DATASETS


class DataFactory:
    def __init__(self, name):
        self._name = name
        self._class = DATASETS[name]

    def seg_dataset(self, **kwargs):
        return self._class(**kwargs)

    @property
    def num_class(self):
        return self._class.NUM_CLASS

    def print_class_label(self, split='train'):
        if split == 'train':
            train_set = self._class(split='train')
            print("Train images: %d" % len(train_set))
            train_class_num = self._print_class_assist([train_set])
            print("Train class label is: %s" % str(train_class_num))
        elif split == 'val':
            val_set = self._class(split='val')
            print("Val images: %d" % len(val_set))
            val_class_num = self._print_class_assist([val_set])
            print("Val class label is: %s" % str(val_class_num))
        else:
            raise RuntimeError(f"Unknown split: {split}")

    @staticmethod
    def _print_class_assist(data_list):
        uniques = []
        for data in data_list:
            bar = tqdm(data)
            for _, (_, mask) in enumerate(bar):
                mask = mask.asnumpy()
                unique = np.unique(mask)
                for v in unique:
                    if v not in uniques:
                        uniques.append(v)
        uniques.sort()
        return uniques
