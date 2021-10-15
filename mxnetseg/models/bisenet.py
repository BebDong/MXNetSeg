# coding=utf-8

from mxnet.gluon import nn
from .base import SegBaseModel, SegBaseResNet
from mxnetseg.utils import MODELS
from .backbone import xception39
from mxnetseg.nn import (FCNHead, ConvModule2d, GlobalFlow, HybridConcurrentIsolate)

__all__ = ['BiSeNetX', 'BiSeNetR']


@MODELS.add_component
class BiSeNetX(SegBaseModel):
    def __init__(self, nclass, backbone='xception39', aux=True, height=None, width=None,
                 base_size=520, crop_size=480, pretrained_base=False, norm_layer=nn.BatchNorm,
                 norm_kwargs=None, **kwargs):
        super(BiSeNetX, self).__init__(nclass, aux, height, width, base_size, crop_size)
        assert backbone == 'xception39', 'support only xception39 as the backbone.'
        pretrained = xception39(pretrained_base, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        with self.name_scope():
            self.conv = pretrained.conv1
            self.max_pool = pretrained.maxpool
            self.layer1 = pretrained.layer1
            self.layer2 = pretrained.layer2
            self.layer3 = pretrained.layer3

            self.head = _BiSeNetHead(nclass, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            if self.aux:
                self.aux_head = HybridConcurrentIsolate()
                self.aux_head.add(FCNHead(nclass, norm_layer=norm_layer, norm_kwargs=norm_kwargs),
                                  FCNHead(nclass, norm_layer=norm_layer, norm_kwargs=norm_kwargs))

    def base_forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.layer1(x)
        c2 = self.layer2(x)
        c3 = self.layer3(c2)
        return c2, c3

    def hybrid_forward(self, F, x, *args, **kwargs):
        c2, c3 = self.base_forward(x)
        outputs = []
        x = self.head(x, c2, c3)
        outputs.append(x)

        if self.aux:
            aux_outs = self.aux_head(c3, c2)
            outputs = outputs + aux_outs
        outputs = [F.contrib.BilinearResize2D(out, **self._up_kwargs) for out in outputs]
        return tuple(outputs)


@MODELS.add_component
class BiSeNetR(SegBaseResNet):
    def __init__(self, nclass, backbone='resnet18', aux=True, height=None, width=None,
                 base_size=520, crop_size=480, pretrained_base=False, norm_layer=nn.BatchNorm,
                 norm_kwargs=None, **kwargs):
        super(BiSeNetR, self).__init__(nclass, aux, backbone, height, width, base_size,
                                       crop_size, pretrained_base, dilate=False,
                                       norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        with self.name_scope():
            self.head = _BiSeNetHead(nclass, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            if self.aux:
                self.aux_head = HybridConcurrentIsolate()
                self.aux_head.add(FCNHead(nclass, norm_layer=norm_layer, norm_kwargs=norm_kwargs),
                                  FCNHead(nclass, norm_layer=norm_layer, norm_kwargs=norm_kwargs))

    def hybrid_forward(self, F, x, *args, **kwargs):
        _, _, c3, c4 = self.base_forward(x)
        outputs = []
        x = self.head(x, c3, c4)
        outputs.append(x)

        if self.aux:
            aux_outs = self.aux_head(c4, c3)
            outputs = outputs + aux_outs
        outputs = [F.contrib.BilinearResize2D(out, **self._up_kwargs) for out in outputs]
        return tuple(outputs)


class _BiSeNetHead(nn.HybridBlock):
    def __init__(self, nclass, norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_BiSeNetHead, self).__init__()
        with self.name_scope():
            self.spatial_path = _SpatialPath(128, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.global_flow = GlobalFlow(128, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.refine_c4 = _ARModule(128, norm_layer, norm_kwargs)
            self.refine_c3 = _ARModule(128, norm_layer, norm_kwargs)
            self.proj = ConvModule2d(128, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.fusion = _FFModule(256, norm_layer, norm_kwargs, reduction=1)
            self.seg = FCNHead(nclass, 256, norm_layer, norm_kwargs, drop_out=.0)

    def hybrid_forward(self, F, x, *args, **kwargs):
        c3, c4 = tuple(args)
        spatial = self.spatial_path(x)
        global_context = self.global_flow(c4)
        global_context = F.contrib.BilinearResize2D(global_context, like=spatial, mode='like')
        refine_c4 = self.refine_c4(c4)
        refine_c4 = F.contrib.BilinearResize2D(refine_c4, like=spatial, mode='like')
        refine_c3 = self.refine_c3(c3)
        refine_c3 = F.contrib.BilinearResize2D(refine_c3, like=spatial, mode='like')
        context = self.proj(global_context + refine_c4 + refine_c3)
        out = self.fusion(spatial, context)
        out = self.seg(out)
        return out


class _SpatialPath(nn.HybridBlock):
    def __init__(self, channels, inter_channels=64, norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_SpatialPath, self).__init__()
        with self.name_scope():
            self.conv7x7 = ConvModule2d(inter_channels, 7, 2, 3, norm_layer=norm_layer,
                                        norm_kwargs=norm_kwargs)
            self.conv3x3_1 = ConvModule2d(inter_channels, 3, 2, 1, norm_layer=norm_layer,
                                          norm_kwargs=norm_kwargs)
            self.conv3x3_2 = ConvModule2d(inter_channels, 3, 2, 1, norm_layer=norm_layer,
                                          norm_kwargs=norm_kwargs)
            self.conv1x1 = ConvModule2d(channels, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv7x7(x)
        x = self.conv3x3_1(x)
        x = self.conv3x3_2(x)
        x = self.conv1x1(x)
        return x


class _ARModule(nn.HybridBlock):
    def __init__(self, channels, norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_ARModule, self).__init__()
        with self.name_scope():
            self.conv3x3 = ConvModule2d(channels, 3, 1, 1, norm_layer=norm_layer,
                                        norm_kwargs=norm_kwargs)
            self.gvp = nn.GlobalAvgPool2D()
            self.conv1x1 = ConvModule2d(channels, 1, in_channels=channels, norm_layer=norm_layer,
                                        norm_kwargs=norm_kwargs, activation='sigmoid')

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv3x3(x)
        score = self.gvp(x)
        score = self.conv1x1(score)
        out = F.broadcast_mul(x, score)
        return out


class _FFModule(nn.HybridBlock):
    def __init__(self, channels, norm_layer=nn.BatchNorm, norm_kwargs=None, reduction=1):
        super(_FFModule, self).__init__()
        with self.name_scope():
            self.proj = ConvModule2d(channels, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.gvp = nn.GlobalAvgPool2D()
            self.conv1x1_1 = ConvModule2d(channels // reduction, 1, norm_layer=norm_layer,
                                          norm_kwargs=norm_kwargs)
            self.conv1x1_2 = ConvModule2d(channels, 1, norm_layer=norm_layer,
                                          norm_kwargs=norm_kwargs, activation='sigmoid')

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = F.concat(x, *args, dim=1)
        x = self.proj(x)
        score = self.gvp(x)
        score = self.conv1x1_1(score)
        score = self.conv1x1_2(score)
        out = F.broadcast_mul(x, score) + x
        return out
