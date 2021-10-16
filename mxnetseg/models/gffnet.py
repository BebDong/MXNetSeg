# coding=utf-8

from mxnet.gluon import nn
from .base import SegBaseResNet
from mxnetseg.utils import MODELS
from mxnetseg.nn import FCNHead, ConvModule2d, PPModule


@MODELS.add_component
class GFFNet(SegBaseResNet):
    """
    Gated fully fusion network.
    Reference:
        X. Li, H. Zhao, et al. “Gated Fully Fusion for Semantic Segmentation,” in AAAI Conference
        on Artificial Intelligence, 2020.
    """

    def __init__(self, nclass, backbone='resnet50', aux=True, height=None, width=None, base_size=520,
                 crop_size=480, pretrained_base=False, norm_layer=nn.BatchNorm, norm_kwargs=None,
                 **kwargs):
        super(GFFNet, self).__init__(nclass, aux, backbone, height, width, base_size, crop_size,
                                     pretrained_base, dilate=True, norm_layer=norm_layer,
                                     norm_kwargs=norm_kwargs)
        with self.name_scope():
            self.head = _GFFHead(nclass, 256, self.stage_channels[3], norm_layer, norm_kwargs)
            if self.aux:
                self.aux_head = FCNHead(nclass, self.stage_channels[2], norm_layer, norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        c1, c2, c3, c4 = self.base_forward(x)
        outputs = []
        x = self.head(c1, c2, c3, c4)
        x = F.contrib.BilinearResize2D(x, **self._up_kwargs)
        outputs.append(x)

        if self.aux:
            aux_out = self.aux_head(c3)
            aux_out = F.contrib.BilinearResize2D(aux_out, **self._up_kwargs)
            outputs.append(aux_out)

        return tuple(outputs)


class _GFFHead(nn.HybridBlock):
    def __init__(self, nclass, channels=256, ppm_in_channels=2048, norm_layer=nn.BatchNorm,
                 norm_kwargs=None):
        super(_GFFHead, self).__init__()
        with self.name_scope():
            self.ppm = nn.HybridSequential()
            self.ppm.add(
                PPModule(ppm_in_channels, norm_layer=norm_layer, norm_kwargs=norm_kwargs),
                ConvModule2d(channels, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            )
            self.conv1x1_1 = ConvModule2d(channels, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv1x1_2 = ConvModule2d(channels, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv1x1_3 = ConvModule2d(channels, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv1x1_4 = ConvModule2d(channels, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.gate_1 = ConvModule2d(channels, 1, use_bias=True, norm_layer=None, activation='sigmoid')
            self.gate_2 = ConvModule2d(channels, 1, use_bias=True, norm_layer=None, activation='sigmoid')
            self.gate_3 = ConvModule2d(channels, 1, use_bias=True, norm_layer=None, activation='sigmoid')
            self.gate_4 = ConvModule2d(channels, 1, use_bias=True, norm_layer=None, activation='sigmoid')
            self.fusion_1 = self._make_fusion(channels, norm_layer, norm_kwargs)
            self.fusion_2 = self._make_fusion(channels, norm_layer, norm_kwargs)
            self.fusion_3 = self._make_fusion(channels, norm_layer, norm_kwargs)
            self.fusion_4 = self._make_fusion(channels, norm_layer, norm_kwargs)
            self.dfp = _DFPModule(nclass, channels, norm_layer, norm_kwargs)

    @staticmethod
    def _make_fusion(channels=256, norm_layer=nn.BatchNorm, norm_kwargs=None):
        fusion = nn.HybridSequential()
        fusion.add(ConvModule2d(channels, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs),
                   ConvModule2d(channels, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        return fusion

    @staticmethod
    def _up_sample(F, feature, like):
        return F.contrib.BilinearResize2D(feature, like=like, mode='like')

    def hybrid_forward(self, F, x, *args, **kwargs):
        c1, c2, c3, c4 = x, args[0], args[1], args[2]
        ppm = self.ppm(c4)
        # learning gates
        gate_1 = self.gate_1(c1)
        gate_2 = self.gate_2(c2)
        gate_3 = self.gate_3(c3)
        gate_4 = self.gate_4(c4)
        # dimension reduction
        c1 = self.conv1x1_1(c1)
        c2 = self.conv1x1_2(c2)
        c3 = self.conv1x1_3(c3)
        c4 = self.conv1x1_4(c4)
        # gated fully fusion
        c1 = c1 + gate_1 * c1 + (1 - gate_1) * (self._up_sample(F, gate_2 * c2, c1) +
                                                self._up_sample(F, gate_3 * c3, c1) +
                                                self._up_sample(F, gate_4 * c4, c1))
        c2 = c2 + gate_2 * c2 + (1 - gate_2) * (self._up_sample(F, gate_1 * c1, c2) +
                                                self._up_sample(F, gate_3 * c3, c2) +
                                                self._up_sample(F, gate_4 * c4, c2))
        c3 = c3 + gate_3 * c3 + (1 - gate_3) * (self._up_sample(F, gate_1 * c1, c3) +
                                                self._up_sample(F, gate_2 * c2, c3) +
                                                self._up_sample(F, gate_4 * c4, c3))
        c4 = c4 + gate_4 * c4 + (1 - gate_4) * (self._up_sample(F, gate_1 * c1, c4) +
                                                self._up_sample(F, gate_2 * c2, c4) +
                                                self._up_sample(F, gate_3 * c3, c4))
        # further fusion of GFF outputs
        c1 = self.fusion_1(c1)
        c2 = self._up_sample(F, self.fusion_2(c2), c1)
        c3 = self._up_sample(F, self.fusion_3(c3), c1)
        c4 = self._up_sample(F, self.fusion_4(c4), c1)
        ppm = self._up_sample(F, ppm, c1)

        return self.dfp(c1, c2, c3, c4, ppm)


class _DFPModule(nn.HybridBlock):
    def __init__(self, nclass, channels, norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_DFPModule, self).__init__()
        with self.name_scope():
            self.blk_4 = ConvModule2d(channels, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.blk_3 = ConvModule2d(channels, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.blk_2 = ConvModule2d(channels, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.blk_1 = ConvModule2d(channels, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.head = FCNHead(nclass, channels * 5, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        c1, c2, c3, c4, ppm = x, args[0], args[1], args[2], args[3]
        out4 = self.blk_4(F.concat(ppm, c4, dim=1))
        out3 = self.blk_3(F.concat(out4, c3, ppm, dim=1))
        out2 = self.blk_2(F.concat(out3, c2, out4, ppm, dim=1))
        out1 = self.blk_1(F.concat(out2, c1, out3, out4, ppm, dim=1))
        return self.head(F.concat(ppm, out4, out3, out2, out1, dim=1))
