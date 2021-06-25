# coding=utf-8

from mxnet.gluon import nn
from .base import SegBaseResNet
from mxnetseg.tools import MODELS
from mxnetseg.nn import FCNHead, AuxHead, ConvBlock, HybridConcurrentSep


@MODELS.add_component
class AttentionToScale(SegBaseResNet):
    """
    ResNet based attention-to-scale model.
    Only support training with two scales of 1.0x and 0.5x.
    Reference: L. C. Chen, Y. Yang, J. Wang, W. Xu, and A. L. Yuille,
        “Attention to Scale: Scale-Aware Semantic Image Segmentation,” in IEEE Conference
         on Computer Vision and Pattern Recognition, 2016, pp. 3640–3649.
    """

    def __init__(self, nclass, backbone='resnet50', aux=True, height=None, width=None,
                 base_size=520, crop_size=480, pretrained_base=True, norm_layer=nn.BatchNorm,
                 norm_kwargs=None, **kwargs):
        super(AttentionToScale, self).__init__(nclass, aux, backbone, height, width, base_size,
                                               crop_size, pretrained_base, dilate=True,
                                               norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        with self.name_scope():
            self.head = _AttentionHead(nclass, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            if self.aux:
                self.aux_head = HybridConcurrentSep()
                self.aux_head.add(
                    AuxHead(nclass, norm_layer=norm_layer, norm_kwargs=norm_kwargs),
                    AuxHead(nclass, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
                )

    def hybrid_forward(self, F, x, *args, **kwargs):
        # 1.0x scale forward
        _, _, _, c4 = self.base_forward(x)
        # 0.5x scale forward
        xh = F.contrib.BilinearResize2D(x,
                                        height=self._up_kwargs['height'] // 2,
                                        width=self._up_kwargs['width'] // 2)
        _, _, _, c4h = self.base_forward(xh)
        # head
        outputs = []
        x = self.head(c4, c4h)
        outputs.append(x)

        if self.aux:
            aux_outs = self.aux_head(c4, c4h)
            outputs = outputs + aux_outs

        outputs = [F.contrib.BilinearResize2D(out, **self._up_kwargs) for out in outputs]
        return tuple(outputs)


class _AttentionHead(nn.HybridBlock):
    """
    Obtain soft attention weights and do weighted summation。
    Here we feed with features from the final stage of ResNet for attention.
    """

    def __init__(self, nclass, norm_layer=nn.BatchNorm, norm_kwargs=None, use_sigmoid=True):
        super(_AttentionHead, self).__init__()
        self.sigmoid = use_sigmoid
        with self.name_scope():
            self.seg_head = FCNHead(nclass, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv3x3 = ConvBlock(512, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            if use_sigmoid:
                self.conv1x1 = nn.Conv2D(1, 1, in_channels=512)
            else:
                self.conv1x1 = nn.Conv2D(2, 1, in_channels=512)

    def hybrid_forward(self, F, x, *args, **kwargs):
        # up-sample for same size
        c4h = F.contrib.BilinearResize2D(args[0], like=x, mode='like')
        # score map
        score_c4 = self.seg_head(x)
        score_c4h = self.seg_head(c4h)
        # obtain soft weights
        weights = F.concat(x, c4h, dim=1)
        weights = self.conv3x3(weights)
        weights = self.conv1x1(weights)
        if self.sigmoid:
            weights = F.sigmoid(weights)
            score_c4 = F.broadcast_mul(weights, score_c4)
            score_c4h = F.broadcast_mul(1 - weights, score_c4h)
            out = score_c4 + score_c4h
        else:
            weights = F.softmax(weights, axis=1)
            # reshape
            weights = F.expand_dims(weights, axis=2)  # (N, 2, 1, H, W)
            score_c4 = F.expand_dims(score_c4, axis=1)
            score_c4h = F.expand_dims(score_c4h, axis=1)
            out = F.concat(score_c4, score_c4h, dim=1)  # (N, 2, nclass, H, W)
            # weighted sum
            out = F.broadcast_mul(weights, out)
            out = F.sum(out, axis=1)
        return out
