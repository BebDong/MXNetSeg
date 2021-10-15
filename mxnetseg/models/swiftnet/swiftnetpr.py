# coding=utf-8

from mxnet.gluon import nn
from ..base import SegBaseResNet
from mxnetseg.nn import FCNHead, AuxHead, ConvModule2d
from mxnetseg.utils import MODELS


@MODELS.add_component
class SwiftResNetPr(SegBaseResNet):
    """
    SwiftNetRN-18 with interleaved pyramid fusion.
    Reference: Orˇ, M., Kreˇ, I., & Bevandi, P. (2019). In Defense of Pre-trained ImageNet
        Architectures for Real-time Semantic Segmentation of Road-driving Images.
        In IEEE Conference on Computer Vision and Pattern Recognition.
    """

    def __init__(self, nclass, backbone='resnet18', aux=False, height=None, width=None,
                 base_size=520, crop_size=480, pretrained_base=False,
                 norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(SwiftResNetPr, self).__init__(nclass, aux, backbone, height, width, base_size,
                                            crop_size, pretrained_base, dilate=False,
                                            norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        with self.name_scope():
            self.head = _SwiftNetHead(nclass, 128, norm_layer, norm_kwargs)
            if self.aux:
                self.aux_head = AuxHead(nclass, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        c1, c2, c3, c4 = self.base_forward(x)
        xh = F.contrib.BilinearResize2D(x, height=self._up_kwargs['height'] // 2,
                                        width=self._up_kwargs['width'] // 2)
        c1h, c2h, c3h, c4h = self.base_forward(xh)
        outputs = []
        out = self.head(c4h, c3h, c2h, c1h, c4, c3, c2, c1)
        out = F.contrib.BilinearResize2D(out, **self._up_kwargs)
        outputs.append(out)
        # auxiliary loss on c3 of 1.0x scale
        if self.aux:
            aux_out = self.aux_head(c3)
            aux_out = F.contrib.BilinearResize2D(aux_out, **self._up_kwargs)
            outputs.append(aux_out)

        return tuple(outputs)


class _SwiftNetHead(nn.HybridBlock):
    """SwiftNet-Pyramid segmentation head"""

    def __init__(self, nclass, capacity=128, norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_SwiftNetHead, self).__init__()
        with self.name_scope():
            self.conv1x1 = ConvModule2d(capacity, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.fusion_32x = _LateralFusion(capacity, norm_layer, norm_kwargs)
            self.fusion_16x = _LateralFusion(capacity, norm_layer, norm_kwargs)
            self.fusion_8x = _LateralFusion(capacity, norm_layer, norm_kwargs)
            self.final = _LateralFusion(capacity, norm_layer, norm_kwargs, is_final=True)
            self.seg_head = FCNHead(nclass, capacity, norm_layer, norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        c3h, c2h, c1h, c4, c3, c2, c1 = args[0], args[1], args[2], args[3], args[4], args[5], args[6]
        c4h = self.conv1x1(x)
        out = self.fusion_32x(c4h, c3h, c4)
        out = self.fusion_16x(out, c2h, c3)
        out = self.fusion_8x(out, c1h, c2)
        out = self.final(out, c1)
        out = self.seg_head(out)
        return out


class _LateralFusion(nn.HybridBlock):
    """
    decoder up-sampling module.
    """

    def __init__(self, capacity, norm_layer=nn.BatchNorm, norm_kwargs=None, is_final=False):
        super(_LateralFusion, self).__init__()
        self.is_final = is_final
        with self.name_scope():
            self.conv1x1 = ConvModule2d(capacity, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv3x3 = ConvModule2d(capacity, 3, 1, 1, norm_layer=norm_layer,
                                        norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        low = args[0] if self.is_final else F.concat(args[0], args[1], dim=1)
        low = self.conv1x1(low)
        high = F.contrib.BilinearResize2D(x, like=low, mode='like')
        out = high + low
        out = self.conv3x3(out)
        return out
