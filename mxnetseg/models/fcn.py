# coding=utf-8

from mxnet.gluon import nn
from mxnetseg.nn import FCNHead
from .base import SegBaseResNet, SegBaseMobileNet
from mxnetseg.utils import MODELS


@MODELS.add_component
class FCNResNet(SegBaseResNet):
    """Fully Convolutional Networks based on ResNet"""

    def __init__(self, nclass, backbone='resnet50', aux=True, height=None, width=None,
                 base_size=520, crop_size=480, pretrained_base=True, dilate=True,
                 norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(FCNResNet, self).__init__(nclass, aux, backbone, height, width,
                                        base_size, crop_size, pretrained_base, dilate,
                                        norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        with self.name_scope():
            self.head = FCNHead(nclass=nclass, in_channels=self.stage_channels[3],
                                norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            if self.aux:
                self.aux_head = FCNHead(nclass, in_channels=self.stage_channels[2],
                                        norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        _, _, c3, c4 = self.base_forward(x)
        outputs = []
        x = self.head(c4)
        x = F.contrib.BilinearResize2D(x, **self._up_kwargs)
        outputs.append(x)

        if self.aux:
            aux_out = self.aux_head(c3)
            aux_out = F.contrib.BilinearResize2D(aux_out, **self._up_kwargs)
            outputs.append(aux_out)
        return tuple(outputs)


@MODELS.add_component
class FCNMobileNet(SegBaseMobileNet):
    """Fully Convolutional Networks based on MobileNet"""

    def __init__(self, nclass, backbone='mobilenet_v2_1_0', aux=True, height=None,
                 width=None, base_size=520, crop_size=480, pretrained_base=True,
                 norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(FCNMobileNet, self).__init__(nclass, aux, backbone, height, width, base_size,
                                           crop_size, pretrained_base, norm_layer=norm_layer,
                                           norm_kwargs=norm_kwargs)
        with self.name_scope():
            self.head = FCNHead(nclass=nclass, in_channels=self.stage_channels[3],
                                norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            if self.aux:
                self.aux_head = FCNHead(nclass=nclass, in_channels=self.stage_channels[2],
                                        norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        _, _, c3, c4 = self.base_forward(x)
        outputs = []
        x = self.head(c4)
        x = F.contrib.BilinearResize2D(x, **self._up_kwargs)
        outputs.append(x)

        if self.aux:
            aux_out = self.aux_head(c3)
            aux_out = F.contrib.BilinearResize2D(aux_out, **self._up_kwargs)
            outputs.append(aux_out)
        return tuple(outputs)
