# coding=utf-8

from mxnet.gluon import nn
from .base import SegBaseResNet
from mxnetseg.nn import FCNHead, PPModule
from mxnetseg.utils import MODELS


@MODELS.add_component
class PSPNet(SegBaseResNet):
    """
    Dilated ResNet50/101/152 based PSPNet.
    Reference: Zhao, H., Shi, J., Qi, X., Wang, X., & Jia, J. (2017).
        Pyramid Scene Parsing Network. In IEEE Conference on Computer Vision and
        Pattern Recognition (pp. 6230–6239). https://doi.org/10.1109/CVPR.2017.660
    """

    def __init__(self, nclass, backbone='resnet50', aux=True, height=None, width=None,
                 base_size=520, crop_size=480, pretrained_base=False,
                 norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(PSPNet, self).__init__(nclass, aux, backbone, height, width, base_size, crop_size,
                                     pretrained_base, dilate=True, norm_layer=norm_layer,
                                     norm_kwargs=norm_kwargs)
        with self.name_scope():
            self.head = _PyramidHead(nclass, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            if self.aux:
                self.aux_head = FCNHead(nclass=nclass, in_channels=1024, norm_layer=norm_layer,
                                        norm_kwargs=norm_kwargs)

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


class _PyramidHead(nn.HybridBlock):
    def __init__(self, nclass, norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_PyramidHead, self).__init__()
        with self.name_scope():
            self.pool = PPModule(2048, norm_layer, norm_kwargs, reduction=4)
            self.seg_head = FCNHead(nclass, 4096, norm_layer, norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.pool(x)
        x = self.seg_head(x)
        return x
