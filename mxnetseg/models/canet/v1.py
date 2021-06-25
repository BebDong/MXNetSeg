# coding=utf-8

from mxnet.gluon import nn
from ..base import SegBaseResNet
from mxnetseg.tools import MODELS
from mxnetseg.nn import FCNHead, ConvBlock, GlobalFlow, DepthSepConvolution


@MODELS.add_component
class CANetv1(SegBaseResNet):
    def __init__(self, nclass, backbone='resnet50', aux=True, height=None, width=None,
                 base_size=520, crop_size=480, pretrained_base=False,
                 norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(CANetv1, self).__init__(nclass, aux, backbone, height, width, base_size, crop_size,
                                      pretrained_base, dilate=True, norm_layer=norm_layer,
                                      norm_kwargs=norm_kwargs)
        with self.name_scope():
            self.head = _CAHead(nclass, norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                height=self._up_kwargs['height'] // 4,
                                width=self._up_kwargs['width'] // 4)
            if self.aux:
                self.aux_head = FCNHead(nclass, self.stage_channels[2], norm_layer, norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        c1, _, c3, c4 = self.base_forward(x)
        outputs = []
        x = self.head(c1, c4)
        x = F.contrib.BilinearResize2D(x, **self._up_kwargs)
        outputs.append(x)

        if self.aux:
            aux_out = self.aux_head(c3)
            aux_out = F.contrib.BilinearResize2D(aux_out, **self._up_kwargs)
            outputs.append(aux_out)

        return tuple(outputs)

    def predict(self, x):
        height, width = x.shape[2:]
        self._up_kwargs['height'] = height
        self._up_kwargs['width'] = width
        self.head.up_kwargs['height'] = height // 4
        self.head.up_kwargs['width'] = width // 4
        self.head.cp1.up_kwargs['height'] = height // 8
        self.head.cp1.up_kwargs['width'] = width // 8
        self.head.cp2.up_kwargs['height'] = height // 8
        self.head.cp2.up_kwargs['width'] = width // 8
        self.head.cp3.up_kwargs['height'] = height // 8
        self.head.cp3.up_kwargs['width'] = width // 8
        self.head.cp4.up_kwargs['height'] = height // 8
        self.head.cp4.up_kwargs['width'] = width // 8
        return self.forward(x)[0]


class _CAHead(nn.HybridBlock):
    def __init__(self, nclass, capacity=512, attention=False, drop=.5, norm_layer=nn.BatchNorm,
                 norm_kwargs=None, height=120, width=120):
        super(_CAHead, self).__init__()
        self.up_kwargs = {'height': height, 'width': width}
        self.attention = attention
        self.gamma = 1.0
        height = height // 2
        width = width // 2
        with self.name_scope():
            # Chained Context Aggregation Module
            self.gp = GlobalFlow(capacity, 2048, norm_layer, norm_kwargs)
            self.cp1 = _ContextFlow(capacity, stride=2, norm_layer=norm_layer,
                                    norm_kwargs=norm_kwargs, height=height, width=width)
            self.cp2 = _ContextFlow(capacity, stride=4, norm_layer=norm_layer,
                                    norm_kwargs=norm_kwargs, height=height, width=width)
            self.cp3 = _ContextFlow(capacity, stride=8, norm_layer=norm_layer,
                                    norm_kwargs=norm_kwargs, height=height, width=width)
            self.cp4 = _ContextFlow(capacity, stride=16, norm_layer=norm_layer,
                                    norm_kwargs=norm_kwargs, height=height, width=width)
            if self.attention:
                self.selection = _FeatureSelection(256, in_channels=capacity, norm_layer=norm_layer,
                                                   norm_kwargs=norm_kwargs)
            else:
                self.proj = ConvBlock(256, 3, 1, 1, in_channels=capacity, norm_layer=norm_layer,
                                      norm_kwargs=norm_kwargs)
            self.drop = nn.Dropout(drop) if drop else None
            # decoder
            self.decoder = ConvBlock(48, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv3x3 = ConvBlock(256, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            # segmentation head
            self.seg_head = FCNHead(nclass, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        c1, c4 = x, args[0]
        # CAM
        context_global = self.gp(c4)
        context_1 = self.cp1(c4, context_global)
        context_2 = self.cp2(c4, context_1)
        context_3 = self.cp3(c4, context_2)
        context_4 = self.cp4(c4, context_3)
        out = self.gamma * context_global + context_1 + context_2 + context_3 + context_4
        # FSM
        if self.attention:
            out = self.selection(out)
        else:
            out = self.proj(out)
        # dropout
        if self.drop:
            out = self.drop(out)
        # decoder
        c1 = self.decoder(c1)
        out = F.contrib.BilinearResize2D(out, **self.up_kwargs)
        out = F.concat(out, c1, dim=1)
        out = self.conv3x3(out)
        # head
        out = self.seg_head(out)
        return out


class _ContextFlow(nn.HybridBlock):
    """
    Context Flow with depth-wise separable convolution.
    """

    def __init__(self, channels, stride, norm_layer=nn.BatchNorm,
                 norm_kwargs=None, height=60, width=60):
        super(_ContextFlow, self).__init__()
        self.stride = stride
        self.up_kwargs = {'height': height, 'width': width}
        with self.name_scope():
            self.conv1 = DepthSepConvolution(channels, 2048 + channels, 3, 1, 1, norm_layer=norm_layer,
                                             norm_kwargs=norm_kwargs)
            self.conv2 = DepthSepConvolution(channels, channels, 3, 1, 1, norm_layer=norm_layer,
                                             norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        upper_input = args[0]
        h, w = self.up_kwargs['height'] // self.stride, self.up_kwargs['width'] // self.stride
        # concat & down-sample
        out = F.concat(x, upper_input, dim=1)
        out = F.contrib.AdaptiveAvgPooling2D(out, output_size=(h, w))
        # Depth Separable Convolution
        out = self.conv1(out)
        out = self.conv2(out)
        # up-sample
        out = F.contrib.BilinearResize2D(out, **self.up_kwargs)
        return out


class _FeatureSelection(nn.HybridBlock):
    """
    Feature Selection Module.
    """

    def __init__(self, channels, in_channels, norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_FeatureSelection, self).__init__()
        with self.name_scope():
            self.conv3x3 = ConvBlock(channels, 3, 1, 1, in_channels=in_channels,
                                     norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.gap = nn.GlobalAvgPool2D()
            self.conv1x1 = ConvBlock(channels, 1, in_channels=channels, norm_layer=norm_layer,
                                     norm_kwargs=norm_kwargs, activation='sigmoid')

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv3x3(x)
        score = self.gap(x)
        score = self.conv1x1(score)
        out = F.broadcast_mul(x, score) + x
        return out


class _ContextFlowShuffle(nn.HybridBlock):
    """
    Context Flow with channel shuffle.
    """

    def __init__(self, channels, stride, groups=4, norm_layer=nn.BatchNorm,
                 norm_kwargs=None, height=60, width=60):
        super(_ContextFlowShuffle, self).__init__()
        self.stride = stride
        self.groups = groups
        self.up_kwargs = {'height': height, 'width': width}
        with self.name_scope():
            self.conv1 = ConvBlock(channels, 3, 1, 1, groups=groups, norm_layer=norm_layer,
                                   norm_kwargs=norm_kwargs, activation='relu')
            self.conv2 = ConvBlock(channels, 3, 1, 1, groups=groups, norm_layer=norm_layer,
                                   norm_kwargs=norm_kwargs, activation='relu')

    def hybrid_forward(self, F, x, *args, **kwargs):
        upper_input = args[0]
        h, w = self.up_kwargs['height'] // self.stride, self.up_kwargs['width'] // self.stride
        # concat & down-sample
        out = F.concat(x, upper_input, dim=1)
        out = F.contrib.AdaptiveAvgPooling2D(out, output_size=(h, w))
        # group convolution & channel shuffle
        out = self.conv1(out)
        out = F.reshape(out, shape=(0, -4, self.groups, -1, -2))
        out = F.transpose(out, axes=(0, 2, 1, 3, 4))
        out = F.reshape(out, shape=(0, -3, -2))
        out = self.conv2(out)
        # up-sample
        out = F.contrib.BilinearResize2D(out, **self.up_kwargs)
        return out
