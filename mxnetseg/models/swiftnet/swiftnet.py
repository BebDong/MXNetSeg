# coding=utf-8

from mxnet.gluon import nn
from ..base import SegBaseResNet
from mxnetseg.nn import FCNHead, PyramidPooling, ConvBlock, LateralFusion
from mxnetseg.tools import MODELS


@MODELS.add_component
class SwiftResNet(SegBaseResNet):
    """
    ResNet based SwiftNet-Single.
    Reference: Orˇ, M., Kreˇ, I., & Bevandi, P. (2019). In Defense of Pre-trained ImageNet Architectures
        for Real-time Semantic Segmentation of Road-driving Images.
        In IEEE Conference on Computer Vision and Pattern Recognition.
    """

    def __init__(self, nclass, backbone='resnet18', aux=True, height=None, width=None,
                 base_size=520, crop_size=480, pretrained_base=False, norm_layer=nn.BatchNorm,
                 norm_kwargs=None, **kwargs):
        super(SwiftResNet, self).__init__(nclass, aux, backbone, height, width, base_size, crop_size,
                                          pretrained_base, dilate=False, norm_layer=norm_layer,
                                          norm_kwargs=norm_kwargs)
        with self.name_scope():
            self.head = _SwiftNetHead(nclass, self.stage_channels[3], norm_layer=norm_layer,
                                      norm_kwargs=norm_kwargs)
            if self.aux:
                self.aux_head = FCNHead(nclass, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        c1, c2, c3, c4 = self.base_forward(x)
        outputs = []
        x = self.head(c4, c3, c2, c1)
        x = F.contrib.BilinearResize2D(x, **self._up_kwargs)
        outputs.append(x)

        if self.aux:
            aux_out = self.aux_head(c3)
            aux_out = F.contrib.BilinearResize2D(aux_out, **self._up_kwargs)
            outputs.append(aux_out)
        return tuple(outputs)


class _SwiftNetHead(nn.HybridBlock):
    """SwiftNet segmentation head"""

    def __init__(self, nclass, in_channels, capacity=256, norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_SwiftNetHead, self).__init__()
        with self.name_scope():
            self.ppool = PyramidPooling(in_channels, norm_layer, norm_kwargs)
            self.conv_c4 = ConvBlock(capacity, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.fusion_c3 = LateralFusion(capacity, norm_layer, norm_kwargs)
            self.fusion_c2 = LateralFusion(capacity, norm_layer, norm_kwargs)
            self.fusion_c1 = LateralFusion(capacity, norm_layer, norm_kwargs)
            self.seg_head = FCNHead(nclass, capacity, norm_layer, norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        c3, c2, c1 = tuple(args)
        c4 = self.ppool(x)
        c4 = self.conv_c4(c4)
        out = self.fusion_c3(c4, c3)
        out = self.fusion_c2(out, c2)
        out = self.fusion_c1(out, c1)
        out = self.seg_head(out)
        return out
