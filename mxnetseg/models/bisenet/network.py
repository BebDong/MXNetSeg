# coding=utf-8

from mxnet.gluon import nn
from ..base import SegBaseResNet
from mxnetseg.nn import FCNHead, ConvBlock, GlobalFlow

__all__ = ['get_bisenet', 'BiSeNet']


class BiSeNet(SegBaseResNet):
    """
    ResNet18 based BiSeNet..
    Reference:
        Yu, C., Wang, J., Peng, C., Gao, C., Yu, G., & Sang, N. (2018).
        BiSeNet: Bilateral segmentation network for real-time semantic segmentation.
        In European Conference on Computer Vision (pp. 334–349).
        https://doi.org/10.1007/978-3-030-01261-8_20
    """

    def __init__(self, nclass, backbone='resnet18', aux=True, height=None, width=None,
                 base_size=520, crop_size=480, pretrained_base=False, norm_layer=nn.BatchNorm,
                 norm_kwargs=None, **kwargs):
        super(BiSeNet, self).__init__(nclass, aux, backbone, height, width, base_size,
                                      crop_size, pretrained_base, dilate=False,
                                      norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        self.head = _BiSeNetHead(nclass, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        if self.aux:
            self.auxlayer = FCNHead(nclass, norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                    drop_out=.0)

    def hybrid_forward(self, F, x, *args, **kwargs):
        _, _, c3, c4 = self.base_forward(x)
        outputs = []
        x = self.head(x, c3, c4)
        x = F.contrib.BilinearResize2D(x, **self._up_kwargs)
        outputs.append(x)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.contrib.BilinearResize2D(auxout, **self._up_kwargs)
            outputs.append(auxout)
        return tuple(outputs)

    def predict(self, x):
        h, w = x.shape[2:]
        self._up_kwargs['height'] = h
        self._up_kwargs['width'] = w
        return self.forward(x)[0]


class _BiSeNetHead(nn.HybridBlock):
    def __init__(self, nclass, norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_BiSeNetHead, self).__init__()
        with self.name_scope():
            self.spatial_path = _SpatialPath(128, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.global_flow = GlobalFlow(128, 512, norm_layer, norm_kwargs)
            self.refine_c4 = _ARModule(128, 512, norm_layer, norm_kwargs)
            self.refine_c3 = _ARModule(128, 256, norm_layer, norm_kwargs)
            self.proj = ConvBlock(128, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
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
    """
    Spatial Path with output stride 8.
    """

    def __init__(self, channels, inter_channels=64, norm_layer=nn.BatchNorm,
                 norm_kwargs=None):
        super(_SpatialPath, self).__init__()
        with self.name_scope():
            self.conv7x7 = ConvBlock(inter_channels, 7, 2, 3, norm_layer=norm_layer,
                                     norm_kwargs=norm_kwargs)
            self.conv3x3_1 = ConvBlock(inter_channels, 3, 2, 1, norm_layer=norm_layer,
                                       norm_kwargs=norm_kwargs)
            self.conv3x3_2 = ConvBlock(inter_channels, 3, 2, 1, norm_layer=norm_layer,
                                       norm_kwargs=norm_kwargs)
            self.conv1x1 = ConvBlock(channels, 1, norm_layer=norm_layer,
                                     norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv7x7(x)
        x = self.conv3x3_1(x)
        x = self.conv3x3_2(x)
        x = self.conv1x1(x)
        return x


class _ARModule(nn.HybridBlock):
    """
    Attention Refinement Module.
    """

    def __init__(self, channels, in_channels=0, norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_ARModule, self).__init__()
        with self.name_scope():
            self.conv3x3 = ConvBlock(channels, 3, 1, 1, in_channels=in_channels,
                                     norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.gvp = nn.GlobalAvgPool2D()
            self.conv1x1 = ConvBlock(channels, 1, in_channels=channels, norm_layer=norm_layer,
                                     norm_kwargs=norm_kwargs, activation='sigmoid')

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv3x3(x)
        score = self.gvp(x)
        score = self.conv1x1(score)
        out = F.broadcast_mul(x, score)
        return out


class _FFModule(nn.HybridBlock):
    """
    Feature Fusion Module.
    """

    def __init__(self, channels, norm_layer=nn.BatchNorm, norm_kwargs=None, reduction=1):
        super(_FFModule, self).__init__()
        with self.name_scope():
            self.proj = ConvBlock(channels, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.gvp = nn.GlobalAvgPool2D()
            self.conv1x1_1 = ConvBlock(channels // reduction, 1, norm_layer=norm_layer,
                                       norm_kwargs=norm_kwargs)
            self.conv1x1_2 = ConvBlock(channels, 1, norm_layer=norm_layer,
                                       norm_kwargs=norm_kwargs, activation='sigmoid')

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = F.concat(x, *args, dim=1)
        x = self.proj(x)
        score = self.gvp(x)
        score = self.conv1x1_1(score)
        score = self.conv1x1_2(score)
        out = F.broadcast_mul(x, score) + x
        return out


def get_bisenet(**kwargs):
    return BiSeNet(**kwargs)
