# coding=utf-8

from mxnet.gluon import nn
from mxnet.gluon.contrib import cnn
from .base import SegBaseResNet
from mxnetseg.utils import MODELS
from mxnetseg.nn import ConvModule2d, FCNHead


@MODELS.add_component
class FaPN(SegBaseResNet):
    """
    Feature-aligned pyramid network.
    Reference:
        Shihua et al., “FaPN: Feature-aligned Pyramid Network for Dense Image Prediction,”
        in International Conference on Computer Vision, 2021.
    """

    def __init__(self, nclass, backbone='resnet50', aux=False, height=None, width=None, base_size=520,
                 crop_size=480, pretrained_base=True, norm_layer=nn.BatchNorm, norm_kwargs=None,
                 **kwargs):
        super(FaPN, self).__init__(nclass, aux, backbone, height, width, base_size, crop_size,
                                   pretrained_base, dilate=False, norm_layer=norm_layer,
                                   norm_kwargs=norm_kwargs)

        with self.name_scope():
            self.head = _FaPNHead(nclass, self.stage_channels, norm_layer=norm_layer,
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


class _FaPNHead(nn.HybridBlock):
    def __init__(self, nclass, in_channels_group, capacity=256, norm_layer=nn.BatchNorm,
                 norm_kwargs=None):
        super(_FaPNHead, self).__init__()
        with self.name_scope():
            self.conv = ConvModule2d(capacity, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.align_c3 = _FAModule(capacity, in_channels_group[2], norm_layer, norm_kwargs)
            self.align_c2 = _FAModule(capacity, in_channels_group[1], norm_layer, norm_kwargs)
            self.align_c1 = _FAModule(capacity, in_channels_group[0], norm_layer, norm_kwargs)
            self.seg = FCNHead(nclass, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        c3, c2, c1 = tuple(args)
        c4 = self.conv(x)
        out = self.align_c3(c4, c3)
        out = self.align_c2(out, c2)
        out = self.align_c1(out, c1)
        out = self.seg(out)
        return out


class _FAModule(nn.HybridBlock):
    def __init__(self, channels, in_channels, norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_FAModule, self).__init__()
        with self.name_scope():
            self.fsm = _FSModule(channels, in_channels, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.offset = ConvModule2d(channels, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.align = cnn.ModulatedDeformableConvolution(channels, 3, 1, 1, num_deformable_group=8)
            self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x, *args, **kwargs):
        low = self.fsm(args[0])
        high = F.contrib.BilinearResize2D(x, like=low, mode='like')
        offset = self.offset(F.concat(low, high, dim=1))
        align = self.align(F.concat(high, offset, dim=1))  # official codes use offset as extra mask
        align = self.relu(align)
        return align + low


class _FSModule(nn.HybridBlock):
    def __init__(self, channels, in_channels, norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_FSModule, self).__init__()
        with self.name_scope():
            self.conv1 = ConvModule2d(in_channels, 1, in_channels=in_channels, norm_layer=norm_layer,
                                      norm_kwargs=norm_kwargs, activation='sigmoid')
            self.gap = nn.GlobalAvgPool2D()
            self.conv2 = ConvModule2d(channels, 1, in_channels=in_channels, norm_layer=None)

    def hybrid_forward(self, F, x, *args, **kwargs):
        score = self.gap(x)
        score = self.conv1(score)
        out = F.broadcast_mul(x, score) + x
        out = self.conv2(out)
        return out
