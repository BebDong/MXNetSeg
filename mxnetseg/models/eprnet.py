# coding=utf-8

from mxnet.gluon import nn
from .base import SegBaseModel
from .backbone import eprnet_cls, eprnet_cls_light
from mxnetseg.tools import MODELS
from mxnetseg.nn import FCNHead


@MODELS.add_component
class EPRNet(SegBaseModel):
    def __init__(self, nclass, height=None, width=None, base_size=520, crop_size=480,
                 pretrained_base=False, norm_layer=nn.BatchNorm, norm_kwargs=None,
                 activation='prelu', drop=0., light=False, **kwargs):
        super(EPRNet, self).__init__(nclass, False, height, width, base_size, crop_size)
        build_backbone = eprnet_cls_light if light else eprnet_cls
        pretrained = build_backbone(pretrained_base, stage_channels=(16, 32, 64, 128),
                                    norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                    activation=activation, drop=drop)
        with self.name_scope():
            self.conv = pretrained.conv
            self.bn = pretrained.bn
            self.act = pretrained.act
            self.layer1 = pretrained.layer1
            self.layer2 = pretrained.layer2
            self.layer3 = pretrained.layer3

            self.head = FCNHead(nclass, pretrained.stage_channels[3], norm_layer=norm_layer,
                                norm_kwargs=norm_kwargs, activation=activation, drop_out=drop)

    def base_forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.base_forward(x)
        outputs = []
        out = self.head(x)
        out = F.contrib.BilinearResize2D(out, **self._up_kwargs)
        outputs.append(out)
        return tuple(outputs)
