# coding=utf-8

from mxnet.gluon import nn
from ..base import SegBaseXception, SegBaseResNet
from mxnetseg.nn import FCNHead, ASPPModule, ConvModule2d
from mxnetseg.utils import MODELS


@MODELS.add_component
class DeepLabv3PlusX(SegBaseXception):
    """
    DeepLab v3+ model.
    Reference: Chen, L. C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018).
        Encoder-decoder with atrous separable convolution for semantic image segmentation.
        In European Conference on Computer Vision (pp. 833–851).
        https://doi.org/10.1007/978-3-030-01234-2_49
    """

    def __init__(self, nclass, backbone='xception65', height=None, width=None, base_size=520,
                 crop_size=480, pretrained_base=True, norm_layer=nn.BatchNorm, norm_kwargs=None,
                 **kwargs):
        super(DeepLabv3PlusX, self).__init__(nclass, False, backbone, height, width, base_size,
                                             crop_size, pretrained_base, dilate=True,
                                             norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        with self.name_scope():
            self.head = _DeepLabHead(nclass, self.stage_channels[2], norm_layer, norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        c1, c3, c4 = self.base_forward(x)
        outputs = []
        x = self.head(c1, c4)
        x = F.contrib.BilinearResize2D(x, **self._up_kwargs)
        outputs.append(x)
        return tuple(outputs)


@MODELS.add_component
class DeepLabv3PlusR(SegBaseResNet):
    """
    ResNet as trunk network.
    """

    def __init__(self, nclass, backbone='resnet50', height=None, width=None, base_size=520,
                 crop_size=480, pretrained_base=False, dilate=True, norm_layer=nn.BatchNorm,
                 norm_kwargs=None, **kwargs):
        super(DeepLabv3PlusR, self).__init__(nclass, False, backbone, height, width, base_size,
                                             crop_size, pretrained_base, dilate,
                                             norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        with self.name_scope():
            self.head = _DeepLabHead(nclass, 2048, norm_layer, norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        c1, _, _, c4 = self.base_forward(x)
        outputs = []
        x = self.head(c1, c4)
        x = F.contrib.BilinearResize2D(x, **self._up_kwargs)
        outputs.append(x)
        return tuple(outputs)


class _DeepLabHead(nn.HybridBlock):
    def __init__(self, nclass, in_channels, norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_DeepLabHead, self).__init__()
        with self.name_scope():
            self.aspp = ASPPModule(256, in_channels, norm_layer, norm_kwargs, rates=(12, 24, 36))
            self.conv_c1 = ConvModule2d(48, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv3x3 = ConvModule2d(256, 3, 1, 1, in_channels=304, norm_layer=norm_layer,
                                        norm_kwargs=norm_kwargs)
            self.drop = nn.Dropout(0.5)
            self.head = FCNHead(nclass, 256, norm_layer, norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        c1, c4 = x, args[0]
        c1 = self.conv_c1(c1)
        out = self.aspp(c4)
        out = F.contrib.BilinearResize2D(out, like=c1, mode='like')
        out = F.concat(c1, out, dim=1)
        out = self.conv3x3(out)
        out = self.drop(out)
        return self.head(out)
