# coding=utf-8

from .segbase import SegBaseModel, ABC
from mxnetseg.utils import data_dir
from gluoncv.model_zoo import mobilenet_v2_0_25, mobilenet_v2_0_5, mobilenet_v2_0_75, mobilenet_v2_1_0

__all__ = ['SegBaseMobileNet']


def _build_backbone(name: str, **kwargs):
    models = {
        'mobilenet_v2_0_25': (mobilenet_v2_0_25, (6, 8, 24, 80)),
        'mobilenet_v2_0_5': (mobilenet_v2_0_5, (12, 16, 48, 160)),
        'mobilenet_v2_0_75': (mobilenet_v2_0_75, (18, 24, 72, 240)),
        'mobilenet_v2_1_0': (mobilenet_v2_1_0, (24, 32, 96, 320)),
    }
    name = name.lower()
    if name not in models.keys():
        raise NotImplementedError(f"Unknown backbone network: {name}")
    model_class, stage_channels = models[name]
    return model_class(**kwargs), stage_channels


class SegBaseMobileNet(SegBaseModel, ABC):
    """
    Backbone MobileNetv2.
    """

    def __init__(self, nclass, aux, backbone='mobilenet_v2_1_0', height=None, width=None,
                 base_size=520, crop_size=480, pretrained_base=True, **kwargs):
        super(SegBaseMobileNet, self).__init__(nclass, aux, height, width, base_size, crop_size)
        pre_trained, channels = _build_backbone(backbone, pretrained=pretrained_base,
                                                root=data_dir(), **kwargs)
        self.stage_channels = channels
        with self.name_scope():
            self.layer1 = pre_trained.features[:6]
            self.layer2 = pre_trained.features[6:9]
            self.layer3 = pre_trained.features[9:16]
            self.layer4 = pre_trained.features[16:20]

    def base_forward(self, x):
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c1, c2, c3, c4
