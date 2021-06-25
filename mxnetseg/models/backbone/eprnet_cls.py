# coding=utf-8

from mxnet.gluon import nn
from mxnet.gluon.contrib.nn import HybridConcurrent
from mxnetseg.nn import Activation, ConvBlock, HybridConcurrentSum
from mxnetseg.tools import validate_checkpoint

__all__ = ['EPRNetCls', 'get_eprnet_cls', 'eprnet_cls', 'eprnet_cls_light']


class EPRNetCls(nn.HybridBlock):
    def __init__(self, light=False, stage_channels=(16, 32, 64, 128), classes=1000,
                 norm_layer=nn.BatchNorm, norm_kwargs=None, activation='prelu',
                 drop=0., **kwargs):
        super(EPRNetCls, self).__init__()
        width1, width2, width3, width4 = tuple(stage_channels)
        self.stage_channels = stage_channels
        with self.name_scope():
            self.conv = nn.Conv2D(channels=width1, kernel_size=3, strides=2,
                                  padding=1, use_bias=False)
            self.bn = norm_layer(**({} if norm_kwargs is None else norm_kwargs))
            self.act = Activation(activation)

            self.layer1 = nn.HybridSequential()
            self.layer1.add(
                _EPRModule(channels=width2, in_channels=width1, atrous_rates=(1, 2, 4),
                           norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                           activation=activation, down_sample=False, light=light),
                _EPRModule(channels=width2, in_channels=width2, atrous_rates=(1, 2, 4),
                           norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                           activation=activation, down_sample=True, light=light))

            self.layer2 = nn.HybridSequential()
            self.layer2.add(
                _EPRModule(channels=width3, in_channels=width2, atrous_rates=(3, 6, 9),
                           norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                           activation=activation, down_sample=False, light=light),
                _EPRModule(channels=width3, in_channels=width3, atrous_rates=(3, 6, 9),
                           norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                           activation=activation, down_sample=False, light=light),
                _EPRModule(channels=width3, in_channels=width3, atrous_rates=(3, 6, 9),
                           norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                           activation=activation, down_sample=False, light=light),
                _EPRModule(channels=width3, in_channels=width3, atrous_rates=(3, 6, 9),
                           norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                           activation=activation, down_sample=True, light=light))

            self.layer3 = nn.HybridSequential()
            self.layer3.add(
                _EPRModule(channels=width4, in_channels=width3, atrous_rates=(7, 13, 19),
                           norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                           activation=activation, down_sample=False, light=light),
                _EPRModule(channels=width4, in_channels=width4, atrous_rates=(13, 25, 37),
                           norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                           activation=activation, down_sample=False, light=light))

            self.avg_pool = nn.GlobalAvgPool2D()
            self.flat = nn.Flatten()
            self.drop = nn.Dropout(drop) if drop > 0. else None
            self.linear = nn.Dense(units=classes)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avg_pool(x)
        x = self.flat(x)
        if self.drop:
            x = self.drop(x)
        x = self.linear(x)

        return x


class _EPRModule(nn.HybridBlock):
    def __init__(self, channels, in_channels, atrous_rates, norm_layer=nn.BatchNorm,
                 norm_kwargs=None, activation='prelu', down_sample=False, light=False):
        super(_EPRModule, self).__init__()
        stride = 2 if down_sample else 1
        with self.name_scope():
            self.pyramid = _MPUnit(channels, atrous_rates, in_channels, norm_layer,
                                   norm_kwargs, activation=activation, light=light)
            self.compact = ConvBlock(channels, 3, stride, 1, in_channels=channels,
                                     norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                     activation=None)

            if (channels != in_channels) or down_sample:
                self.skip = nn.Conv2D(channels, kernel_size=1, strides=stride,
                                      use_bias=False, in_channels=in_channels)
                self.skip_bn = norm_layer(**({} if norm_kwargs is None else norm_kwargs))
            else:
                self.skip = None

            self.act = Activation(activation)

    def hybrid_forward(self, F, x, *args, **kwargs):
        residual = x
        out = self.pyramid(x)
        out = self.compact(out)
        if self.skip:
            residual = self.skip(residual)
            residual = self.skip_bn(residual)
        out = out + residual
        return self.act(out)


class _MPUnit(nn.HybridBlock):
    def __init__(self, channels, atrous_rates, in_channels, norm_layer=nn.BatchNorm,
                 norm_kwargs=None, activation='prelu', light=False, **kwargs):
        super(_MPUnit, self).__init__()
        with self.name_scope():
            self.concurrent = HybridConcurrent(axis=1) if not light else HybridConcurrentSum(axis=1)
            for i in range(len(atrous_rates)):
                rate = atrous_rates[i]
                self.concurrent.add(ConvBlock(channels, 3, 1, padding=rate, dilation=rate,
                                              groups=in_channels, in_channels=in_channels,
                                              norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                              activation=activation))
            if not light:
                self.concurrent.add(ConvBlock(channels, 1, in_channels=in_channels,
                                              norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                              activation=activation))
                self.conv1x1 = ConvBlock(channels, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                         activation=activation)
            else:
                self.conv1x1 = None

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.concurrent(x)
        if self.conv1x1:
            x = self.conv1x1(x)
        return x


def get_eprnet_cls(light, pretrained=False, ckpt_name=None, **kwargs):
    model = EPRNetCls(light, **kwargs)
    if pretrained:
        model_weight = validate_checkpoint('EPRNetCls', ckpt_name)
        model.load_parameters(model_weight)
    return model


def eprnet_cls(pretrained=False, **kwargs):
    return get_eprnet_cls(False, pretrained, ckpt_name='eprnetcls_imagenet.params', **kwargs)


def eprnet_cls_light(pretrained=False, **kwargs):
    return get_eprnet_cls(True, pretrained, ckpt_name='eprnetcls_light_imagenet.params', **kwargs)
