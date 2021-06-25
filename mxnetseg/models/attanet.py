# coding=utf-8

from .base import SegBaseResNet
from mxnet.gluon import nn
from mxnetseg.nn import ConvBlock, FCNHead
from mxnetseg.tools import MODELS


@MODELS.add_component
class AttaNet(SegBaseResNet):
    """
    Attention-augmented network.
    Reference:
         Q. Song, K. Mei, and R. Huang, “AttaNet: Attention-Augmented Network for Fast and
         Accurate Scene Parsing,” in AAAI Conference on Artificial Intelligence, 2021.
    """

    def __init__(self, nclass, backbone='resnet18', aux=True, height=None, width=None,
                 base_size=520, crop_size=480, pretrained_base=False, norm_layer=nn.BatchNorm,
                 norm_kwargs=None, **kwargs):
        super(AttaNet, self).__init__(nclass, aux, backbone, height, width, base_size,
                                      crop_size, pretrained_base, dilate=False,
                                      norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        with self.name_scope():
            self.head = _AttaNetHead(nclass, norm_layer, norm_kwargs)
            if self.aux:
                self.aux_head = FCNHead(nclass, self.stage_channels[2], norm_layer, norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        _, _, c3, c4 = self.base_forward(x)
        outputs = []
        x = self.head(c4, c3)
        x = F.contrib.BilinearResize2D(x, **self._up_kwargs)
        outputs.append(x)

        if self.aux:
            aux_out = self.aux_head(c3)
            aux_out = F.contrib.BilinearResize2D(aux_out, **self._up_kwargs)
            outputs.append(aux_out)
        return tuple(outputs)


class _AttaNetHead(nn.HybridBlock):
    def __init__(self, nclass, norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_AttaNetHead, self).__init__()
        with self.name_scope():
            self.afm = _AttentionFusionModule(128, norm_layer, norm_kwargs)
            self.conv3x3 = ConvBlock(128, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.sam = _StripAttentionModule(128, norm_layer, norm_kwargs)
            self.seg = FCNHead(nclass, 128, norm_layer, norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        c4, c3 = x, args[0]
        out = self.afm(c4, c3)
        out = self.conv3x3(out)
        out = self.sam(out)
        out = self.seg(out)
        return out


class _AttentionFusionModule(nn.HybridBlock):
    def __init__(self, channels=128, norm_layer=None, norm_kwargs=None):
        super(_AttentionFusionModule, self).__init__()
        with self.name_scope():
            self.conv3x3_high = ConvBlock(channels, 3, 1, 1, norm_layer=norm_layer,
                                          norm_kwargs=norm_kwargs)
            self.conv3x3_low = ConvBlock(channels, 3, 1, 1, norm_layer=norm_layer,
                                         norm_kwargs=norm_kwargs)
            self.conv1x1_1 = ConvBlock(channels, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv1x1_2 = ConvBlock(channels, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                       activation='sigmoid')
            self.gap = nn.GlobalAvgPool2D()

    def hybrid_forward(self, F, x, *args, **kwargs):
        low = self.conv3x3_low(args[0])
        high = F.contrib.BilinearResize2D(x, like=low, mode='like')
        weight = F.concat(high, low, dim=1)
        weight = self.conv1x1_1(weight)
        weight = self.gap(weight)
        weight = self.conv1x1_2(weight)

        x = self.conv3x3_high(x)
        x = F.broadcast_mul(x, weight)
        low = F.broadcast_mul(low, 1 - weight)
        x = F.contrib.BilinearResize2D(x, like=low, mode='like')

        return x + low


class _StripAttentionModule(nn.HybridBlock):
    def __init__(self, in_channels, norm_layer=None, norm_kwargs=None, reduction=2):
        super(_StripAttentionModule, self).__init__()
        with self.name_scope():
            self.query_conv = ConvBlock(in_channels // reduction, 1, in_channels=in_channels,
                                        norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.key_conv = ConvBlock(in_channels // reduction, 1, in_channels=in_channels,
                                      norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.value_conv = ConvBlock(in_channels, 1, in_channels=in_channels,
                                        norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        query = F.reshape(self.query_conv(x), shape=(0, 0, -1))  # NC(HW)
        key = F.mean(self.key_conv(x), axis=2)  # NCW
        affinity = F.batch_dot(query, key, transpose_a=True)  # N(HW)W
        affinity = F.softmax(affinity)
        value = F.mean(self.value_conv(x), axis=2)  # NCW
        out = F.batch_dot(value, affinity, transpose_b=True)  # NC(HW)
        out = F.reshape_like(out, x, lhs_begin=2, lhs_end=None, rhs_begin=2, rhs_end=None)
        return out + x
