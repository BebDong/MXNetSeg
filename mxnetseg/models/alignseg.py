# coding=utf-8

from mxnet.gluon import nn
from .base import SegBaseResNet
from mxnetseg.utils import MODELS
from mxnetseg.nn import FCNHead, ConvModule2d


@MODELS.add_component
class AlignSeg(SegBaseResNet):
    """
    Feature-aligned segmentation network.
    Reference:
        Z. Huang, Y. Wei, X. Wang, H. Shi, W. Liu, and T. S. Huang, “AlignSeg: Feature-Aligned
        Segmentation Networks,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 8828, 2021.
    """

    def __init__(self, nclass, backbone='resnet50', aux=True, height=None, width=None, base_size=520,
                 crop_size=480, pretrained_base=False, norm_layer=nn.BatchNorm, norm_kwargs=None,
                 **kwargs):
        super(AlignSeg, self).__init__(nclass, aux, backbone, height, width, base_size, crop_size,
                                       pretrained_base, dilate=False, norm_layer=norm_layer,
                                       norm_kwargs=norm_kwargs)
        with self.name_scope():
            self.head = _AlignHead(nclass, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            if self.aux:
                self.aux_head = FCNHead(nclass, self.stage_channels[3], norm_layer, norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        c1, c2, c3, c4 = self.base_forward(x)
        outputs = []
        x = self.head(c1, c2, c3, c4)
        x = F.contrib.BilinearResize2D(x, **self._up_kwargs)
        outputs.append(x)

        if self.aux:
            aux_out = self.aux_head(c4)
            aux_out = F.contrib.BilinearResize2D(aux_out, **self._up_kwargs)
            outputs.append(aux_out)

        return tuple(outputs)


class _AlignHead(nn.HybridBlock):
    def __init__(self, nclass, channels=256, norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_AlignHead, self).__init__()
        with self.name_scope():
            self.align_cm = _AlignCM(channels, norm_layer, norm_kwargs)
            self.align_1 = _AlignBlock(channels, norm_layer, norm_kwargs)
            self.align_2 = _AlignBlock(channels, norm_layer, norm_kwargs)
            self.align_3 = _AlignBlock(channels, norm_layer, norm_kwargs)
            self.align_4 = _AlignBlock(channels, norm_layer, norm_kwargs)
            self.rcb = _RCBlock(channels, norm_layer, norm_kwargs)
            self.head = FCNHead(nclass, channels, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        f1, f2, f3 = args[0], args[1], args[2]
        f4 = self.align_cm(f3)
        a1 = self.align_1(f1, x)
        a2 = self.align_2(f2, a1)
        a3 = self.align_3(f3, a2)
        out = self.align_4(f4, a3)
        out = self.rcb(out)
        return self.head(out)


class _AlignBlock(nn.HybridBlock):
    def __init__(self, channels=256, norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_AlignBlock, self).__init__()
        with self.name_scope():
            self.rcb_1 = _RCBlock(channels, norm_layer, norm_kwargs)
            self.rcb_2 = _RCBlock(channels, norm_layer, norm_kwargs)
            self.align = _AlignFA(channels, norm_layer, norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        high = self.rcb_1(x)
        low = self.rcb_2(args[0])
        return self.align(high, low)


class _RCBlock(nn.HybridBlock):
    def __init__(self, channels=256, norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_RCBlock, self).__init__()
        with self.name_scope():
            self.conv1x1 = nn.Conv2D(channels, 1, use_bias=False)
            self.conv3x3_1 = ConvModule2d(channels // 4, 3, 1, 1, norm_layer=norm_layer,
                                          norm_kwargs=norm_kwargs)
            self.conv3x3_2 = nn.Conv2D(channels, 3, 1, 1, use_bias=False)
            self.bn = norm_layer(**({} if norm_kwargs is None else norm_kwargs))
            self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv1x1(x)
        residual = x
        x = self.conv3x3_1(x)
        x = self.conv3x3_2(x)
        x = self.relu(self.bn(x + residual))
        return x


class _AlignFA(nn.HybridBlock):
    def __init__(self, channels=256, norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_AlignFA, self).__init__()
        with self.name_scope():
            self.delta = nn.HybridSequential()
            self.delta.add(
                ConvModule2d(channels, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs),
                nn.Conv2D(4, 3, 1, 1, use_bias=False)
            )

    def hybrid_forward(self, F, x, *args, **kwargs):
        high_res = args[0]
        low_res = F.contrib.BilinearResize2D(x, like=high_res, mode='like')
        grid = self.delta(F.concat(low_res, high_res, dim=1))
        grid_high, grid_low = F.split_v2(grid, 2, axis=1)
        grid_high = F.GridGenerator(grid_high, transform_type='warp')
        grid_low = F.GridGenerator(grid_low, transform_type='warp')
        high_res = F.BilinearSampler(high_res, grid_high)
        low_res = F.BilinearSampler(low_res, grid_low)
        return high_res + low_res


class _AlignCM(nn.HybridBlock):
    def __init__(self, channels=256, norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_AlignCM, self).__init__()
        with self.name_scope():
            self.conv1x1 = ConvModule2d(channels, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv3x3 = ConvModule2d(channels, 3, 1, 1, norm_layer=norm_layer,
                                        norm_kwargs=norm_kwargs)
            self.delta = nn.HybridSequential()
            self.delta.add(
                ConvModule2d(channels, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs),
                nn.Conv2D(2, 3, 1, 1, use_bias=False)
            )

    def hybrid_forward(self, F, x, *args, **kwargs):
        high = F.contrib.BilinearResize2D(self.conv1x1(F.contrib.AdaptiveAvgPooling2D(x, output_size=3)),
                                          like=x, mode='like')
        low = self.conv3x3(x)
        grid = self.delta(F.concat(high, low, dim=1))
        grid = F.GridGenerator(grid, transform_type='warp')
        high = F.BilinearSampler(high, grid)
        return F.concat(high, low, dim=1)
