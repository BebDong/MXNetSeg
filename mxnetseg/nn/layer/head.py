# coding=utf-8

from mxnet.gluon import nn
from .naive import ConvBlock

__all__ = ['FCNHead', 'AuxHead', 'DUpsampling']


class FCNHead(nn.HybridBlock):
    """
    segmentation head for FCN based network.
    """

    def __init__(self, nclass, in_channels=0, norm_layer=nn.BatchNorm, norm_kwargs=None,
                 activation='relu', drop_out=.1):
        super(FCNHead, self).__init__()
        inter_channels = 256 if (in_channels == 0) or (in_channels > 256) else in_channels
        with self.name_scope():
            self.conv1 = ConvBlock(inter_channels, 3, 1, 1, in_channels=in_channels,
                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                   activation=activation)

            self.drop = nn.Dropout(drop_out) if drop_out else None
            self.conv2 = nn.Conv2D(nclass, 1, in_channels=inter_channels)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv1(x)
        if self.drop:
            x = self.drop(x)
        x = self.conv2(x)

        return x


class AuxHead(nn.HybridBlock):
    """
    auxiliary segmentation head.
    """

    def __init__(self, nclass, in_channels=0, norm_layer=nn.BatchNorm, norm_kwargs=None,
                 activation='relu'):
        super(AuxHead, self).__init__()
        inter_channels = 256 if (in_channels == 0) or (in_channels > 256) else in_channels
        with self.name_scope():
            self.conv1 = ConvBlock(inter_channels, 1, in_channels=in_channels,
                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                   activation=activation)
            self.conv2 = nn.Conv2D(nclass, 1, in_channels=inter_channels)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DUpsampling(nn.HybridBlock):
    """
    DUpsampling operation.
    Reference: Tian, Z., He, T., Shen, C., & Yan, Y. (2019). Decoders Matter for Semantic Segmentation:
        Data-Dependent Decoding Enables Flexible Feature Aggregation.
        In IEEE Conference on Computer Vision and Pattern Recognition (pp. 3126–3135).
    Adapted From: https://github.com/LinZhuoChen/DUpsampling
    """

    def __init__(self, nclass, in_channels, scale, input_height=60, input_width=60):
        super(DUpsampling, self).__init__()
        self.scale = scale
        self.size_kwargs = {'channels': nclass * scale * scale,
                            'height': input_height, 'width': input_width}
        with self.name_scope():
            self.w_matrix = nn.Conv2D(nclass * scale * scale, 1, use_bias=False, in_channels=in_channels)
            # self.p_matrix = nn.Conv2D(in_channels, 1, use_bias=False, in_channels=nclass * scale * scale)

    def hybrid_forward(self, F, x, *args, **kwargs):
        # NCHW
        x = self.w_matrix(x)
        channels = self.size_kwargs['channels']
        height = self.size_kwargs['height']
        width = self.size_kwargs['width']
        # NWHC
        x = F.transpose(x, axes=(0, 3, 2, 1))
        # NW(H*scale)(C/scale)
        x = F.reshape(x, shape=(-1, width, height * self.scale, int(channels / self.scale)))
        # N(H*scale)W(C/scale)
        x = F.transpose(x, axes=(0, 2, 1, 3))
        # N(H*scale)(W*scale)(C/(scale**2))
        x = F.reshape(x, shape=(
            -1, height * self.scale, width * self.scale, int(channels / (self.scale * self.scale))))
        # N(C/(scale**2))(H*scale)(W*scale)
        x = F.transpose(x, axes=(0, 3, 1, 2))

        return x
