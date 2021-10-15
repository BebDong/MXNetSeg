# coding=utf-8

from typing import Any
from mxnet.gluon import nn
from .nonlinear import Activation

__all__ = ['ConvModule2d', 'DepthwiseSeparableConv2d', 'FactorizedConv2d',
           'HybridConcurrentSum', 'HybridConcurrentIsolate', 'HybridSequentialUpscale']


class ConvModule2d(nn.HybridBlock):
    def __init__(self, channels, kernel_size, strides: Any = 1, padding: Any = 0, dilation=1,
                 groups=1, in_channels=0, use_bias=False, norm_layer=nn.BatchNorm, norm_kwargs=None,
                 activation='relu', **kwargs):
        super(ConvModule2d, self).__init__()
        with self.name_scope():
            self.conv = nn.Conv2D(channels, kernel_size, strides, padding, dilation, groups,
                                  in_channels=in_channels, activation=None, use_bias=use_bias,
                                  **kwargs)
            self.bn = norm_layer(**({} if norm_kwargs is None else norm_kwargs)) if norm_layer else None
            self.act = Activation(activation) if activation else None

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.act:
            x = self.act(x)
        return x


class DepthwiseSeparableConv2d(nn.HybridBlock):
    def __init__(self, channels, in_channels, kernel_size, strides=1, padding=0, dilation=1,
                 use_bias=False, norm_layer=nn.BatchNorm, norm_kwargs=None, activation='relu',
                 pattern='mobilenet'):
        super(DepthwiseSeparableConv2d, self).__init__()
        assert pattern in ('xception', 'mobilenet')
        with self.name_scope():
            if pattern == 'xception':
                self.depthwise = nn.Conv2D(in_channels, kernel_size, strides, padding, dilation,
                                           groups=in_channels, in_channels=in_channels,
                                           activation=None, use_bias=use_bias)
            else:
                self.depthwise = ConvModule2d(in_channels, kernel_size, strides, padding, dilation,
                                              groups=in_channels, in_channels=in_channels,
                                              use_bias=use_bias, norm_layer=norm_layer,
                                              norm_kwargs=norm_kwargs, activation=activation)
            self.pointwise = ConvModule2d(channels, 1, in_channels=in_channels, use_bias=use_bias,
                                          norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                          activation=activation)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class FactorizedConv2d(nn.HybridBlock):
    def __init__(self, channels, kernel_size, strides=1, padding=0, in_channels=0, use_bias=False,
                 norm_layer=nn.BatchNorm, norm_kwargs=None, activation='relu'):
        super(FactorizedConv2d, self).__init__()
        with self.name_scope():
            self.conv1 = ConvModule2d(channels, (kernel_size, 1), (strides, 1), (padding, 0),
                                      in_channels=in_channels, use_bias=use_bias,
                                      norm_layer=None, activation=activation)
            self.conv2 = ConvModule2d(channels, (1, kernel_size), (1, strides), (0, padding),
                                      in_channels=channels, use_bias=use_bias,
                                      norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                      activation=activation)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class HybridConcurrentSum(nn.HybridSequential):
    """
    Lays multiple `HybridBlock` concurrently and produce the output by element-sum all
    children blocks' outputs.
    """

    def __init__(self, prefix=None, params=None):
        super(HybridConcurrentSum, self).__init__(prefix=prefix, params=params)

    def hybrid_forward(self, F, x):
        out = []
        for blk in self._children.values():
            out.append(blk(x))
        out = F.ElementWiseSum(*out)
        return out


class HybridConcurrentIsolate(nn.HybridSequential):
    """
    Lays multiple `HybridBlock` concurrently but produce the output separately with one
    children block corresponding to one input.
    """

    def __init__(self, prefix=None, params=None):
        super(HybridConcurrentIsolate, self).__init__(prefix=prefix, params=params)

    def hybrid_forward(self, F, *inputs):
        out = []
        for x, blk in zip(inputs, self._children.values()):
            out.append(blk(x))
        return out


class HybridSequentialUpscale(nn.HybridSequential):
    """
    Lays multiple `HybridBlock` sequentially where each children block is followed by 2x up.
    """

    def __init__(self, prefix=None, params=None):
        super(HybridSequentialUpscale, self).__init__(prefix=prefix, params=params)

    def hybrid_forward(self, F, x):
        for blk in self._children.values():
            x = blk(x)
            x = F.contrib.BilinearResize2D(x, scale_height=2., scale_width=2.)
        return x
