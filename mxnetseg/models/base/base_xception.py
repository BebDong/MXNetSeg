# coding=utf-8

from .segbase import SegBaseModel, ABC
from mxnetseg.tools import data_dir
from gluoncv.model_zoo import get_xcetption

__all__ = ['SegBaseXception']


class SegBaseXception(SegBaseModel, ABC):
    def __init__(self, nclass, aux, backbone='xception65', height=None, width=None,
                 base_size=520, crop_size=480, pretrained_base=True, dilate=True,
                 **kwargs):
        super(SegBaseXception, self).__init__(nclass, aux, height, width, base_size, crop_size)
        assert backbone == 'xception65', f"Unknown backbone {backbone}"
        output_stride = 8 if dilate else 32
        pre_trained = get_xcetption(pretrained_base, output_stride=output_stride,
                                    root=data_dir(), **kwargs)
        self.stage_channels = (128, 728, 2048)
        with self.name_scope():
            # base network
            self.conv1 = pre_trained.conv1
            self.bn1 = pre_trained.bn1
            self.relu = pre_trained.relu
            self.conv2 = pre_trained.conv2
            self.bn2 = pre_trained.bn2
            self.block1 = pre_trained.block1
            self.block2 = pre_trained.block2
            self.block3 = pre_trained.block3
            # Middle flow
            self.midflow = pre_trained.midflow
            # Exit flow
            self.block20 = pre_trained.block20
            self.conv3 = pre_trained.conv3
            self.bn3 = pre_trained.bn3
            self.conv4 = pre_trained.conv4
            self.bn4 = pre_trained.bn4
            self.conv5 = pre_trained.conv5
            self.bn5 = pre_trained.bn5

    def base_forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        # add relu here
        x = self.relu(x)
        low_level_feat = x
        x = self.block2(x)
        x = self.block3(x)
        # Middle flow
        x = self.midflow(x)
        mid_level_feat = x
        # Exit flow
        x = self.block20(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        return low_level_feat, mid_level_feat, x
