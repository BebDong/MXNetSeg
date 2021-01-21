# coding=utf-8

from mxnet import init
from mxnet.gluon import nn
from mxnet.gluon.contrib.nn import HybridConcurrent

from .naive import ConvBlock

__all__ = ['GlobalFlow', 'PyramidPooling', 'DenseASPPBlock', 'SeEModule', 'SelfAttention',
           'LateralFusion', 'GCN', 'ASPP']


class GlobalFlow(nn.HybridBlock):
    """
    Global Average Pooling to obtain global receptive field.
    """

    def __init__(self, channels, in_channels=0, norm_layer=nn.BatchNorm, norm_kwargs=None, ):
        super(GlobalFlow, self).__init__()
        with self.name_scope():
            self.gap = nn.GlobalAvgPool2D()
            self.conv1x1 = ConvBlock(channels, 1, in_channels=in_channels, norm_layer=norm_layer,
                                     norm_kwargs=norm_kwargs, activation='relu')

    def hybrid_forward(self, F, x, *args, **kwargs):
        out = self.gap(x)
        out = self.conv1x1(out)
        out = F.contrib.BilinearResize2D(out, like=x, mode='like')
        return out


class PyramidPooling(nn.HybridBlock):
    """
    Pyramid Pooling Module.
    Reference: Zhao, H., Shi, J., Qi, X., Wang, X., & Jia, J. (2017).
        Pyramid Scene Parsing Network. In IEEE Conference on Computer Vision and
        Pattern Recognition (pp. 6230–6239). https://doi.org/10.1109/CVPR.2017.660
    """

    def __init__(self, in_channels, norm_layer=nn.BatchNorm, norm_kwargs=None,
                 activation='relu', reduction=4):
        super(PyramidPooling, self).__init__()
        with self.name_scope():
            self.conv1 = ConvBlock(in_channels // reduction, 1, in_channels=in_channels,
                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                   activation=activation)
            self.conv2 = ConvBlock(in_channels // reduction, 1, in_channels=in_channels,
                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                   activation=activation)
            self.conv3 = ConvBlock(in_channels // reduction, 1, in_channels=in_channels,
                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                   activation=activation)
            self.conv4 = ConvBlock(in_channels // reduction, 1, in_channels=in_channels,
                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                   activation=activation)

    def hybrid_forward(self, F, x, *args, **kwargs):
        feature1 = F.contrib.BilinearResize2D(self.conv1(F.contrib.AdaptiveAvgPooling2D
                                                         (x, output_size=1)),
                                              like=x, mode='like')
        feature2 = F.contrib.BilinearResize2D(self.conv2(F.contrib.AdaptiveAvgPooling2D
                                                         (x, output_size=2)),
                                              like=x, mode='like')
        feature3 = F.contrib.BilinearResize2D(self.conv3(F.contrib.AdaptiveAvgPooling2D
                                                         (x, output_size=3)),
                                              like=x, mode='like')
        feature4 = F.contrib.BilinearResize2D(self.conv4(F.contrib.AdaptiveAvgPooling2D
                                                         (x, output_size=6)),
                                              like=x, mode='like')
        return F.concat(x, feature1, feature2, feature3, feature4, dim=1)


class GCN(nn.HybridBlock):
    """
    Global Convolution Network.
    Employing {1xr,rx1} and {rx1,1xr} to approximate the rxr convolution.
    Reference:
        Peng, Chao, et al. "Large Kernel Matters--Improve Semantic Segmentation by
        Global Convolutional Network." Proceedings of the IEEE conference on computer
        vision and pattern recognition. 2017.
    """

    def __init__(self, channels, k_size, in_channels=0):
        super(GCN, self).__init__()
        pad = int((k_size - 1) / 2)
        with self.name_scope():
            self.conv_l1 = nn.Conv2D(channels, kernel_size=(k_size, 1), padding=(pad, 0),
                                     in_channels=in_channels)
            self.conv_l2 = nn.Conv2D(channels, kernel_size=(1, k_size), padding=(0, pad),
                                     in_channels=channels)
            self.conv_r1 = nn.Conv2D(channels, kernel_size=(1, k_size), padding=(0, pad),
                                     in_channels=in_channels)
            self.conv_r2 = nn.Conv2D(channels, kernel_size=(k_size, 1), padding=(pad, 0),
                                     in_channels=channels)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        return x_l + x_r


class ASPP(nn.HybridBlock):
    """
    Atrous Spatial Pyramid Pooling.
    Reference:
        Chen, L.-C., Papandreou, G., Schroff, F., & Adam, H. (2017).
        Rethinking Atrous Convolution for Semantic Image Segmentation.
        https://doi.org/10.1002/cncr.24278
    """

    def __init__(self, channels=256, in_channels=0, norm_layer=nn.BatchNorm, norm_kwargs=None,
                 rates=(6, 12, 18), drop=.5, pool_branch=True):
        super(ASPP, self).__init__()
        with self.name_scope():
            self.branches = HybridConcurrent(axis=1)
            self.branches.add(ConvBlock(channels, 1, in_channels=in_channels, norm_layer=norm_layer,
                                        norm_kwargs=norm_kwargs))
            for rate in rates:
                self.branches.add(ConvBlock(channels, 3, padding=rate, dilation=rate,
                                            in_channels=in_channels, norm_layer=norm_layer,
                                            norm_kwargs=norm_kwargs))
            if pool_branch:
                self.branches.add(GlobalFlow(channels, in_channels, norm_layer, norm_kwargs))

            self.projection = ConvBlock(channels, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.drop = nn.Dropout(drop) if drop else None

    def hybrid_forward(self, F, x, *args, **kwargs):
        out = self.branches(x)
        out = self.projection(out)
        if self.drop:
            out = self.drop(out)
        return out


class DenseASPPBlock(nn.HybridBlock):
    """
    DenseASPP block.
        input channels = 2048 with dilated ResNet50/101/152
        output channels of 1x1 transition layer = 1024
        output channels of each dilate 3x3 Conv = 256
    Reference: Zhang, C., Li, Z., Yu, K., Yang, M., & Yang, K. (2018).
        DenseASPP for Semantic Segmentation in Street Scenes.
        In IEEE Conference on Computer Vision and Pattern Recognition (pp. 3684–3692).
        IEEE. https://doi.org/10.1109/cvpr.2018.00388
    """

    def __init__(self, channels=256, in_channels=2048, norm_layer=nn.BatchNorm,
                 norm_kwargs=None, atrous_rates=(3, 6, 12, 18, 24)):
        super(DenseASPPBlock, self).__init__()
        rate1, rate2, rate3, rate4, rate5 = tuple(atrous_rates)
        out_dilate = in_channels // 8
        out_transition = in_channels // 2
        with self.name_scope():
            self.dilate1 = ConvBlock(out_dilate, 3, padding=rate1, dilation=rate1,
                                     norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.trans1 = ConvBlock(out_transition, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.dilate2 = ConvBlock(out_dilate, 3, padding=rate2, dilation=rate2,
                                     norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.trans2 = ConvBlock(out_transition, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.dilate3 = ConvBlock(out_dilate, 3, padding=rate3, dilation=rate3,
                                     norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.trans3 = ConvBlock(out_transition, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.dilate4 = ConvBlock(out_dilate, 3, padding=rate4, dilation=rate4,
                                     norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.trans4 = ConvBlock(out_transition, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.dilate5 = ConvBlock(out_dilate, 3, padding=rate5, dilation=rate5,
                                     norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.projection = ConvBlock(channels, 1, in_channels=3328, norm_layer=norm_layer,
                                        norm_kwargs=norm_kwargs, activation='relu')
            self.drop = nn.Dropout(0.5)

    def hybrid_forward(self, F, x, *args, **kwargs):
        out1 = self.dilate1(x)

        out2 = F.concat(x, out1, dim=1)
        out2 = self.trans1(out2)
        out2 = self.dilate2(out2)

        out3 = F.concat(x, out1, out2, dim=1)
        out3 = self.trans2(out3)
        out3 = self.dilate3(out3)

        out4 = F.concat(x, out1, out2, out3, dim=1)
        out4 = self.trans3(out4)
        out4 = self.dilate4(out4)

        out5 = F.concat(x, out1, out2, out3, out4, dim=1)
        out5 = self.trans4(out5)
        out5 = self.dilate5(out5)

        out = F.concat(x, out1, out2, out3, out4, out5, dim=1)
        out = self.projection(out)
        return self.drop(out)


class SeEModule(nn.HybridBlock):
    """
    Semantic Enhancement Module. i.e. modified ASPP block.
    Not employ Depth-wise separable convolution here since it saves little parameters.
    Reference:
        Pang, Yanwei, et al. "Towards bridging semantic gap to improve semantic segmentation."
        Proceedings of the IEEE International Conference on Computer Vision. 2019.
    """

    def __init__(self, channels=128, atrous_rates=(1, 2, 4, 8), norm_layer=nn.BatchNorm,
                 norm_kwargs=None, full_sample=False):
        super(SeEModule, self).__init__()
        rate1, rate2, rate3, rate4 = tuple(atrous_rates)
        with self.name_scope():
            self.branch1 = self._make_branch(rate1, channels, norm_layer, norm_kwargs, full_sample)
            self.branch2 = self._make_branch(rate2, channels, norm_layer, norm_kwargs, full_sample)
            self.branch3 = self._make_branch(rate3, channels, norm_layer, norm_kwargs, full_sample)
            self.branch4 = self._make_branch(rate4, channels, norm_layer, norm_kwargs, full_sample)
            self.proj = ConvBlock(channels, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = F.concat(x, out1, out2, out3, out4, dim=1)
        return self.proj(out)

    @staticmethod
    def _make_branch(dilation, channels, norm_layer, norm_kwargs, full_sample):
        branch = nn.HybridSequential()
        branch.add(ConvBlock(channels, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        if full_sample:
            branch.add(GCN(channels, 3, in_channels=channels))
        branch.add(ConvBlock(channels, 3, padding=dilation, dilation=dilation,
                             norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        return branch


class LateralFusion(nn.HybridBlock):
    """
    Lateral fusion adopted in U-shape models.
    Reference:
        Kreso I, Segvic S, Krapac J. Ladder-style densenets for semantic segmentation of
        large natural images[C]//Proceedings of the IEEE International Conference on Computer
        Vision Workshops. 2017: 238-245.

        Orsic M, Kreso I, Bevandic P, et al. In defense of pre-trained imagenet architectures
         for real-time semantic segmentation of road-driving images[C]//Proceedings of the
         IEEE conference on computer vision and pattern recognition. 2019: 12607-12616.
    """

    def __init__(self, capacity, norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(LateralFusion, self).__init__()
        with self.name_scope():
            self.conv1x1 = ConvBlock(capacity, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv3x3 = ConvBlock(capacity, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        high, low = x, args[0]
        high = F.contrib.BilinearResize2D(high, like=low, mode='like')
        low = self.conv1x1(low)
        out = high + low
        return self.conv3x3(out)


class SelfAttention(nn.HybridBlock):
    """
    Self-attention.
    References: Zhang, H., Goodfellow, I., Metaxas, D., & Odena, A. (2018).
        Self-attention generative adversarial networks. arXiv preprint arXiv:1805.08318.
    """

    def __init__(self, in_channels, reduction=8):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        with self.name_scope():
            self.query_conv = nn.Conv2D(in_channels // reduction, 1, in_channels=in_channels)
            self.key_conv = nn.Conv2D(in_channels // reduction, 1, in_channels=in_channels)
            self.value_conv = nn.Conv2D(in_channels, 1, in_channels=in_channels)
            self.gamma = self.params.get('gamma', shape=(1,), init=init.Zero())

    def hybrid_forward(self, F, x, *args, **kwargs):
        """
        input:
            feature maps with NCHW shape
        returns:
            out: self-attention feature maps + input feature maps with shape NCHW
            attention: self-attention score with shape N(HW)(HW).
        """
        gamma = kwargs['gamma']
        query = F.reshape(self.query_conv(x), shape=(0, 0, -1))  # NC(HW)
        key = F.reshape(self.key_conv(x), shape=(0, 0, -1))  # NC(HW)
        energy = F.batch_dot(query, key, transpose_a=True)  # N(HW)(HW)
        attention = F.softmax(energy)
        value = F.reshape(self.value_conv(x), shape=(0, 0, -1))  # NC(HW)
        out = F.batch_dot(value, attention, transpose_b=True)  # NC(HW)
        out = F.reshape_like(out, x, lhs_begin=2, lhs_end=None, rhs_begin=2, rhs_end=None)
        out = F.broadcast_mul(gamma, out) + x
        return out
