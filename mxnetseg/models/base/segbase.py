# coding=utf-8

from abc import ABC, abstractmethod
from mxnet.gluon.nn import HybridBlock

__all__ = ['ABC', 'SegBaseModel']


class SegBaseModel(HybridBlock, ABC):
    """base model for semantic segmentation"""

    def __init__(self, nclass: int, aux: bool, height: int = None, width: int = None,
                 base_size: int = 520, crop_size: int = 480, symbolize: bool = True):
        super(SegBaseModel, self).__init__()
        self.nclass = nclass
        self.aux = aux
        height = height if height is not None else crop_size
        width = width if width is not None else crop_size
        self._up_kwargs = {'height': height, 'width': width}
        self.base_size = base_size
        self.crop_size = crop_size

        self.symbolize = symbolize

    @abstractmethod
    def base_forward(self, x):
        """define feed forward of backbone network"""
        pass

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)[0]

    def predict(self, x):
        """predict with raw images"""
        h, w = x.shape[2:]
        self._up_kwargs['height'] = h
        self._up_kwargs['width'] = w
        return self.forward(x)[0]
