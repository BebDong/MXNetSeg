# coding=utf-8

from mxnet.gluon import nn
from .base import SegBaseResNet
from mxnetseg.nn import ASPP, FCNHead, ConvBlock
from mxnetseg.tools import MODELS


@MODELS.add_component
class ACFNet(SegBaseResNet):
    """
    Attentional Class Feature Network.
    Reference: F. Zhang et al., “ACFNet: Attentional Class Feature Network for
        Semantic Segmentation,” in IEEE International Conference on Computer Vision, 2019.
    """

    def __init__(self, nclass, backbone='resnet50', aux=False, height=None, width=None,
                 base_size=520, crop_size=480, pretrained_base=True, norm_layer=nn.BatchNorm,
                 norm_kwargs=None, **kwargs):
        super(ACFNet, self).__init__(nclass, aux, backbone, height, width, base_size,
                                     crop_size, pretrained_base, dilate=True,
                                     norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        with self.name_scope():
            self.head = _ACFHead(nclass, self.stage_channels[3], norm_layer, norm_kwargs)
            if self.aux:
                self.aux_head = FCNHead(nclass, self.stage_channels[2], norm_layer, norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        _, _, c3, c4 = self.base_forward(x)
        outputs = []
        out, coarse = self.head(c4)
        out = F.contrib.BilinearResize2D(out, **self._up_kwargs)
        outputs.append(out)

        if self.aux:
            coarse_out = F.contrib.BilinearResize2D(coarse, **self._up_kwargs)
            outputs.append(coarse_out)
            aux_out = self.aux_head(c3)
            aux_out = F.contrib.BilinearResize2D(aux_out, **self._up_kwargs)
            outputs.append(aux_out)

        return tuple(outputs)


class _ACFHead(nn.HybridBlock):
    def __init__(self, nclass, in_channels, norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_ACFHead, self).__init__()
        with self.name_scope():
            self.aspp = ASPP(512, in_channels, norm_layer, norm_kwargs,
                             rates=(12, 24, 36), pool_branch=False)
            self.coarse_head = FCNHead(nclass, 512, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.acf = _ACFModule(512, 512, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.head = FCNHead(nclass, 1024, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        aspp_feat = self.aspp(x)
        coarse = self.coarse_head(aspp_feat)
        acf_feat = self.acf(aspp_feat, coarse)
        out = F.concat(aspp_feat, acf_feat)
        out = self.head(out)
        return out, coarse


class _ACFModule(nn.HybridBlock):
    def __init__(self, channels, in_channels, norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_ACFModule, self).__init__()
        with self.name_scope():
            self.conv_1 = ConvBlock(channels, 1, in_channels=in_channels,
                                    norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv_2 = ConvBlock(channels, 1, in_channels=in_channels,
                                    norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv_1(x)
        feat = F.reshape(x, shape=(0, 0, -1))  # NC'(HW)
        coarse = F.reshape(args[0], shape=(0, 0, -1))  # NC(HW)
        energy = F.batch_dot(coarse, feat, transpose_b=True)  # NCC'
        energy_new = F.max(energy, -1, True).broadcast_like(energy) - energy
        attention = F.softmax(energy_new)
        out = F.batch_dot(attention, coarse, transpose_a=True)  # NC'(HW)
        out = F.reshape_like(out, x, lhs_begin=2, lhs_end=None, rhs_begin=2, rhs_end=None)
        out = self.conv_2(out)
        return out
