# coding=utf-8

from mxnet.gluon import nn
from .base import SegBaseDenseNet
from mxnetseg.nn import FCNHead, ConvModule2d, LateralFusion
from mxnetseg.utils import MODELS


@MODELS.add_component
class LadderDenseNet(SegBaseDenseNet):
    """
    ladder-style DenseNet.
    Reference: Krešo, I., Šegvić, S., & Krapac, J. (2018). Ladder-Style DenseNets for
        Semantic Segmentation of Large Natural Images.
        In IEEE International Conference on Computer Vision (pp. 238–245).
        https://doi.org/10.1109/ICCVW.2017.37
    """

    def __init__(self, nclass, backbone='densenet121', aux=True, height=None, width=None,
                 base_size=520, crop_size=480, pretrained_base=False,
                 norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(LadderDenseNet, self).__init__(nclass, aux, backbone, height, width, base_size,
                                             crop_size, pretrained_base, norm_layer=norm_layer,
                                             norm_kwargs=norm_kwargs)
        decoder_capacity = 128
        with self.name_scope():
            self.head = _LadderHead(nclass, decoder_capacity, norm_layer, norm_kwargs)
            if self.aux:
                self.aux_head = FCNHead(nclass, in_channels=decoder_capacity, norm_layer=norm_layer,
                                        norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        c1, c2, c3, c4 = self.base_forward(x)
        outputs = []
        x, c4 = self.head(c4, c3, c2, c1)
        x = F.contrib.BilinearResize2D(x, **self._up_kwargs)
        outputs.append(x)

        if self.aux:
            aux_out = self.aux_head(c4)
            aux_out = F.contrib.BilinearResize2D(aux_out, **self._up_kwargs)
            outputs.append(aux_out)
        return tuple(outputs)


class _LadderHead(nn.HybridBlock):
    """decoder for LadderResNet"""

    def __init__(self, nclass, decoder_capacity, norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_LadderHead, self).__init__()
        with self.name_scope():
            self.conv_c4 = ConvModule2d(decoder_capacity, 1, norm_layer=norm_layer,
                                        norm_kwargs=norm_kwargs)
            self.fusion_c3 = LateralFusion(decoder_capacity, norm_layer, norm_kwargs)
            self.fusion_c2 = LateralFusion(decoder_capacity, norm_layer, norm_kwargs)
            self.fusion_c1 = LateralFusion(decoder_capacity, norm_layer, norm_kwargs)
            self.seg_head = FCNHead(nclass, decoder_capacity, norm_layer, norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        c3, c2, c1 = tuple(args)
        c4 = self.conv_c4(x)
        out = self.fusion_c3(c4, c3)
        out = self.fusion_c2(out, c2)
        out = self.fusion_c1(out, c1)
        out = self.seg_head(out)
        return out, c4
