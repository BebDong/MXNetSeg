# coding=utf-8

from mxnet.gluon import nn
from ..base import SegBaseResNet
from mxnetseg.nn import FCNHead, ASPPModule
from mxnetseg.utils import MODELS


@MODELS.add_component
class DeepLabv3(SegBaseResNet):
    """
    ResNet18/34/50/101/152 based DeepLab_v3 framework.
    Reference: Chen, L.-C., Papandreou, G., Schroff, F., & Adam, H. (2017).
        Rethinking Atrous Convolution for Semantic Image Segmentation.
        https://doi.org/10.1002/cncr.24278
    """

    def __init__(self, nclass, backbone='resnet50', aux=True, height=None, width=None,
                 base_size=520, crop_size=480, pretrained_base=False, dilate=True,
                 norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(DeepLabv3, self).__init__(nclass, aux, backbone, height, width, base_size,
                                        crop_size, pretrained_base, dilate,
                                        norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        self.output_stride = 8 if dilate else 32
        with self.name_scope():
            self.head = _DeepLabHead(nclass, self.stage_channels[3], norm_layer, norm_kwargs)
            if self.aux:
                self.aux_head = FCNHead(nclass, self.stage_channels[2], norm_layer, norm_kwargs)

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


class _DeepLabHead(nn.HybridBlock):
    def __init__(self, nclass, in_channels, norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_DeepLabHead, self).__init__()
        with self.name_scope():
            self.aspp = ASPPModule(256, in_channels, norm_layer, norm_kwargs, rates=(6, 12, 18))
            self.head = FCNHead(nclass, 256, norm_layer, norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.aspp(x)
        return self.head(x)
