# coding=utf-8
# Adapted from: https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/segbase.py

from .segbase import SegBaseModel, ABC
from mxnetseg.tools import data_dir
from gluoncv.model_zoo import resnet18_v1b, resnet34_v1b, resnet50_v1s, resnet101_v1s, resnet152_v1s
from gluoncv.model_zoo.resnest import resnest14, resnest26, resnest50, resnest101, resnest200, resnest269

__all__ = ['SegBaseResNet']


def _build_backbone(name: str, **kwargs):
    models = {
        'resnet18': (resnet18_v1b, (64, 128, 256, 512)),
        'resnet34': (resnet34_v1b, (64, 128, 256, 512)),
        'resnet50': (resnet50_v1s, (256, 512, 1024, 2048)),
        'resnet101': (resnet101_v1s, (256, 512, 1024, 2048)),
        'resnet152': (resnet152_v1s, (256, 512, 1024, 2048)),
        'resnest14': (resnest14, (256, 512, 1024, 2048)),
        'resnest26': (resnest26, (256, 512, 1024, 2048)),
        'resnest50': (resnest50, (256, 512, 1024, 2048)),
        'resnest101': (resnest101, (256, 512, 1024, 2048)),
        'resnest200': (resnest200, (256, 512, 1024, 2048)),
        'resnest269': (resnest269, (256, 512, 1024, 2048)),
    }
    name = name.lower()
    if name not in models.keys():
        raise NotImplementedError(f"Unknown backbone network: {name}")
    model_class, stage_channels = models[name]
    return model_class(**kwargs), stage_channels


class SegBaseResNet(SegBaseModel, ABC):
    def __init__(self, nclass, aux, backbone='resnet50', height=None, width=None,
                 base_size=520, crop_size=480, pretrained_base=True, dilate=True,
                 **kwargs):
        super(SegBaseResNet, self).__init__(nclass, aux, height, width, base_size, crop_size)
        pre_trained, channels = _build_backbone(backbone, pretrained=pretrained_base,
                                                root=data_dir(), dilated=dilate, **kwargs)
        self.stage_channels = channels
        with self.name_scope():
            self.conv1 = pre_trained.conv1
            self.bn1 = pre_trained.bn1
            self.relu = pre_trained.relu
            self.maxpool = pre_trained.maxpool
            self.layer1 = pre_trained.layer1
            self.layer2 = pre_trained.layer2
            self.layer3 = pre_trained.layer3
            self.layer4 = pre_trained.layer4

    def base_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c1, c2, c3, c4
