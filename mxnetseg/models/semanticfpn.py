# coding=utf-8

from mxnet.gluon import nn
from .base import SegBaseResNet
from mxnetseg.nn import ConvBlock, LateralFusion, UpscaleLayer
from mxnetseg.tools import MODELS


@MODELS.add_component
class SemanticFPN(SegBaseResNet):
    """
    PanopticFPN with only semantic segmentation branch.
    Reference:  [1] A. Kirillov, R. Girshick, K. He, and P. Dollár, “Panoptic Feature
        Pyramid Networks,” in IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    """

    def __init__(self, nclass, backbone='resnet50', height=None, width=None, base_size=520,
                 crop_size=480, pretrained_base=True, norm_layer=nn.BatchNorm, norm_kwargs=None,
                 **kwargs):
        super(SemanticFPN, self).__init__(nclass, False, backbone, height, width, base_size,
                                          crop_size, pretrained_base, False, norm_layer=norm_layer,
                                          norm_kwargs=norm_kwargs)
        with self.name_scope():
            self.head = _SemanticHead(nclass, norm_layer, norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        c1, c2, c3, c4 = self.base_forward(x)
        outputs = []
        x = self.head(c4, c3, c2, c1)
        x = F.contrib.BilinearResize2D(x, **self._up_kwargs)
        outputs.append(x)
        return tuple(outputs)


class _SemanticHead(nn.HybridBlock):
    def __init__(self, nclass, norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_SemanticHead, self).__init__()
        with self.name_scope():
            self.fpn = _FPNBranch(256, norm_layer, norm_kwargs)
            self.semantic = _SemanticBranch(128, norm_layer, norm_kwargs)
            self.seg = nn.Conv2D(nclass, 1, in_channels=128)

    def hybrid_forward(self, F, x, *args, **kwargs):
        c3, c2, c1 = tuple(args)
        p5, p4, p3, p2 = self.fpn(x, c3, c2, c1)
        out = self.semantic(p5, p4, p3, p2)
        out = self.seg(out)
        return out


class _SemanticBranch(nn.HybridBlock):
    def __init__(self, capacity=128, norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_SemanticBranch, self).__init__()
        with self.name_scope():
            self.scale_p5 = self._make_layer(3, capacity, norm_layer, norm_kwargs)
            self.scale_p4 = self._make_layer(2, capacity, norm_layer, norm_kwargs)
            self.scale_p3 = self._make_layer(1, capacity, norm_layer, norm_kwargs)
            self.scale_p2 = self._make_layer(0, capacity, norm_layer, norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        p4, p3, p2 = tuple(args)
        p5 = self.scale_p5(x)
        p4 = self.scale_p4(p4)
        p3 = self.scale_p3(p3)
        p2 = self.scale_p2(p2)
        out = p5 + p4 + p3 + p2
        return out

    @staticmethod
    def _make_layer(stages, channels, norm_layer=nn.BatchNorm, norm_kwargs=None):
        # scale: 1/4 --> 1/4
        if stages == 0:
            layer = ConvBlock(channels, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            return layer
        # n = 2 ^ stages
        # scale: 1/(4 * n) --> 1/4
        layer = UpscaleLayer()
        for _ in range(stages):
            layer.add(ConvBlock(channels, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        return layer


class _FPNBranch(nn.HybridBlock):
    def __init__(self, capacity=256, norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_FPNBranch, self).__init__()
        with self.name_scope():
            self.conv = ConvBlock(capacity, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.lateral16x = LateralFusion(capacity, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.lateral8x = LateralFusion(capacity, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.lateral4x = LateralFusion(capacity, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        c3, c2, c1 = tuple(args)
        p5 = self.conv(x)
        p4 = self.lateral16x(p5, c3)
        p3 = self.lateral8x(p4, c2)
        p2 = self.lateral4x(p3, c1)
        return p5, p4, p3, p2
