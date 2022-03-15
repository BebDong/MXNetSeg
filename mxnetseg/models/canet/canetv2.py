# coding=utf-8

import math
from mxnet.gluon import nn
from ..base import SegBaseResNet
from mxnetseg.utils import MODELS
from mxnetseg.nn import FCNHead, ConvModule2d, GlobalFlow


@MODELS.add_component
class CANetv2(SegBaseResNet):
    def __init__(self, nclass, backbone='resnet50', aux=False, height=None, width=None,
                 base_size=520, crop_size=480, pretrained_base=False, dilate=True,
                 norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(CANetv2, self).__init__(nclass, aux, backbone, height, width, base_size, crop_size,
                                      pretrained_base, dilate=dilate, norm_layer=norm_layer,
                                      norm_kwargs=norm_kwargs)
        with self.name_scope():
            self.head = _CANetHead(nclass, channels=512, norm_layer=norm_layer,
                                   norm_kwargs=norm_kwargs)

            if self.aux:
                self.aux_head = FCNHead(nclass, self.stage_channels[2], norm_layer, norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        _, _, c3, c4 = self.base_forward(x)
        outputs = []
        out = self.head(c4, c3)
        out = F.contrib.BilinearResize2D(out, **self._up_kwargs)
        outputs.append(out)

        if self.aux:
            aux_out = self.aux_head(c3)
            aux_out = F.contrib.BilinearResize2D(aux_out, **self._up_kwargs)
            outputs.append(aux_out)

        return tuple(outputs)


class _CANetHead(nn.HybridBlock):
    def __init__(self, nclass, channels, norm_layer=None, norm_kwargs=None, drop=.0):
        super(_CANetHead, self).__init__()
        self.num_ed = 3
        with self.name_scope():
            # compression
            self.comp_c4 = ConvModule2d(channels, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.comp_c3 = ConvModule2d(channels, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.gap = GlobalFlow(48, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            # context encoding: parallel encoder-decoders
            self.ed1 = _EncoderDecoder(2, channels, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.ed2 = _EncoderDecoder(4, channels, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.ed3 = _EncoderDecoder(8, channels, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            # gated fusion: gated sum of encoder-decoders and low-level features
            self.conv1 = ConvModule2d(channels, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv2 = ConvModule2d(self.num_ed + 1, 1, norm_layer=norm_layer,
                                      norm_kwargs=norm_kwargs, activation=None)
            # segmentation head
            self.drop = nn.Dropout(drop) if drop else None
            self.seg = FCNHead(nclass, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        c4 = self.comp_c4(x)
        c3 = self.comp_c3(args[0])

        gap = self.gap(x)
        mid1 = self.ed1(c4)
        mid2 = self.ed2(c4)
        mid3 = self.ed3(c4)

        score = F.concat(mid1, mid2, mid3, c3, dim=1)
        score = self.conv1(score)
        score = self.conv2(score)
        score = F.softmax(score, axis=1)
        score = F.expand_dims(score, axis=2)

        c3 = F.expand_dims(c3, axis=1)
        mid1 = F.expand_dims(mid1, axis=1)
        mid2 = F.expand_dims(mid2, axis=1)
        mid3 = F.expand_dims(mid3, axis=1)
        mid = F.concat(mid1, mid2, mid3, c3, dim=1)
        out = F.broadcast_mul(mid, score)
        out = F.sum(out, axis=1)
        if self.drop:
            out = self.drop(out)

        out = F.concat(out, gap, dim=1)
        out = self.seg(out)
        return out


class _EncoderDecoder(nn.HybridBlock):
    def __init__(self, scale, channels, norm_layer=None, norm_kwargs=None):
        super(_EncoderDecoder, self).__init__()
        num_blocks = int(math.log(scale, 2))
        with self.name_scope():
            # encoder
            self.encoder = nn.HybridSequential()
            for i in range(num_blocks):
                self.encoder.add(_Bottleneck(channels, norm_layer, norm_kwargs))
            self.conv3x3 = ConvModule2d(channels, 3, 1, 1, norm_layer=norm_layer,
                                        norm_kwargs=norm_kwargs)
            # decoder
            self.conv1x1 = ConvModule2d(channels, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        out = self.encoder(x)
        out = self.conv3x3(out)
        out = F.contrib.BilinearResize2D(out, like=x, mode='like')
        out = self.conv1x1(out)
        return out


class _Bottleneck(nn.HybridBlock):
    def __init__(self, channels, norm_layer=None, norm_kwargs=None):
        super(_Bottleneck, self).__init__()
        inner_channels = 128
        with self.name_scope():
            self.conv1 = ConvModule2d(inner_channels, 1, norm_layer=norm_layer,
                                      norm_kwargs=norm_kwargs)
            self.conv2 = ConvModule2d(inner_channels, 3, 2, 1, norm_layer=norm_layer,
                                      norm_kwargs=norm_kwargs)
            self.conv3 = ConvModule2d(channels, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv4 = ConvModule2d(channels, 1, 2, norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                      activation=None)

    def hybrid_forward(self, F, x, *args, **kwargs):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + self.conv4(residual)
        return out
