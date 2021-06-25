# coding=utf-8

from mxnet import init
from mxnet.gluon import nn
from .base import SegBaseModel
from .backbone import vit_large_16
from mxnetseg.tools import MODELS
from mxnetseg.nn import ConvBlock, HybridConcurrentSep


@MODELS.add_component
class SETR(SegBaseModel):
    def __init__(self, nclass, aux=True, backbone='vit_large_16', height=None, width=None,
                 base_size=520, crop_size=480, pretrained_base=False, norm_layer=nn.BatchNorm,
                 norm_kwargs=None, decoder='mla', layer_norm_eps=1e-6, **kwargs):
        super(SETR, self).__init__(nclass, aux, height, width, base_size, crop_size, symbolize=False)
        assert backbone == 'vit_large_16', 'only support vit_large_16 for now'
        assert decoder in ('naive', 'pup', 'mla'), 'decoder must be any of (naive, pup, mla)'
        encoder = vit_large_16(pretrained_base, img_size=crop_size, classes=1000)
        self.stride = crop_size // encoder.patch_size
        with self.name_scope():
            # embedding
            self.patch_embed = encoder.patch_embed
            self.pos_embed = self.params.get('pos_embed',
                                             shape=(1, self.patch_embed.num_patches, encoder.embed_dim),
                                             init=init.Zero())
            self.embed_dropout = encoder.embed_dropout
            # encoder
            self.blocks = encoder.blocks
            # decoder
            self.out_indices, self.layer_norms, head = self._build_decoder(decoder, layer_norm_eps)
            self.head = head(nclass, aux, norm_layer, norm_kwargs)

    @staticmethod
    def _build_decoder(decoder, layer_norm_eps):
        if decoder == 'naive':
            out_indices = (10, 15, 20, 24)
            head = _NaiveHead
        elif decoder == 'pup':
            out_indices = (10, 15, 20, 24)
            head = _PUPHead
        else:
            out_indices = (6, 12, 18, 24)
            head = _MLAHead
        out_indices = tuple([i - 1 for i in out_indices])
        layer_norms = HybridConcurrentSep()
        for i in range(len(out_indices)):
            layer_norms.add(nn.LayerNorm(epsilon=layer_norm_eps))
        return out_indices, layer_norms, head

    def base_forward(self, x):
        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                outputs.append(x)
        return outputs

    def hybrid_forward(self, F, x, *args, **kwargs):
        # embedding
        pos_embed = kwargs['pos_embed']
        x = self.patch_embed(x)
        x = F.broadcast_add(x, pos_embed)
        if self.embed_dropout:
            x = self.embed_dropout(x)
        # encoder
        outputs = self.base_forward(x)
        outputs = self.layer_norms(*outputs)
        outputs.reverse()
        # decoder
        outputs = [F.transpose(F.reshape(out, shape=(0, self.stride, self.stride, -1)),
                               axes=(0, 3, 1, 2)) for out in outputs]
        outputs = self.head(*outputs)
        outputs = [F.contrib.BilinearResize2D(out, **self._up_kwargs) for out in outputs]
        return tuple(outputs)


class _NaiveHead(nn.HybridBlock):
    def __init__(self, nclass, aux, norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_NaiveHead, self).__init__()
        self.aux = aux
        with self.name_scope():
            self.head = _SegHead(nclass, norm_layer, norm_kwargs)
            if self.aux:
                self.aux_head = HybridConcurrentSep()
                self.aux_head.add(
                    _SegHead(nclass, norm_layer, norm_kwargs),
                    _SegHead(nclass, norm_layer, norm_kwargs),
                    _SegHead(nclass, norm_layer, norm_kwargs)
                )

    def hybrid_forward(self, F, x, *args, **kwargs):
        outputs = []
        out = self.head(x)
        outputs.append(out)
        if self.aux:
            aux_outs = self.aux_head(*args)
            outputs = outputs + aux_outs
        return tuple(outputs)


class _PUPHead(nn.HybridBlock):
    def __init__(self, nclass, aux, norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_PUPHead, self).__init__()
        self.aux = aux
        with self.name_scope():
            self.conv0 = ConvBlock(256, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv1 = ConvBlock(256, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv2 = ConvBlock(256, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv3 = ConvBlock(256, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv4 = ConvBlock(nclass, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            if self.aux:
                self.aux_head = HybridConcurrentSep()
                self.aux_head.add(
                    _SegHead(nclass, norm_layer, norm_kwargs),
                    _SegHead(nclass, norm_layer, norm_kwargs),
                    _SegHead(nclass, norm_layer, norm_kwargs),
                    _SegHead(nclass, norm_layer, norm_kwargs)
                )

    def hybrid_forward(self, F, x, *args, **kwargs):
        _, _, h, w = x.shape
        outputs = []
        out = self.conv0(x)
        out = F.contrib.BilinearResize2D(out, height=h * 2, width=h * 2)
        out = self.conv1(out)
        out = F.contrib.BilinearResize2D(out, height=h * 4, width=h * 4)
        out = self.conv2(out)
        out = F.contrib.BilinearResize2D(out, height=h * 8, width=h * 8)
        out = self.conv4(self.conv3(out))
        outputs.append(out)
        if self.aux:
            aux_outs = self.aux_head(x, *args)
            outputs = outputs + aux_outs
        return tuple(outputs)


class _MLAHead(nn.HybridBlock):
    def __init__(self, nclass, aux, norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_MLAHead, self).__init__()
        self.aux = aux
        with self.name_scope():
            # top-down aggregation
            self.conv1x1_p5 = ConvBlock(256, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv1x1_p4 = ConvBlock(256, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv1x1_p3 = ConvBlock(256, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv1x1_p2 = ConvBlock(256, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv3x3_p5 = ConvBlock(256, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv3x3_p4 = ConvBlock(256, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv3x3_p3 = ConvBlock(256, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv3x3_p2 = ConvBlock(256, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            # segmentation head
            self.head5 = nn.HybridSequential()
            self.head5.add(
                ConvBlock(128, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs),
                ConvBlock(128, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            self.head4 = nn.HybridSequential()
            self.head4.add(
                ConvBlock(128, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs),
                ConvBlock(128, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            self.head3 = nn.HybridSequential()
            self.head3.add(
                ConvBlock(128, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs),
                ConvBlock(128, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            self.head2 = nn.HybridSequential()
            self.head2.add(
                ConvBlock(128, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs),
                ConvBlock(128, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            self.head = nn.Conv2D(nclass, 1, in_channels=128 * 4)
            if self.aux:
                self.aux_head = HybridConcurrentSep()
                self.aux_head.add(
                    _SegHead(nclass, norm_layer, norm_kwargs),
                    _SegHead(nclass, norm_layer, norm_kwargs),
                    _SegHead(nclass, norm_layer, norm_kwargs),
                    _SegHead(nclass, norm_layer, norm_kwargs)
                )

    def hybrid_forward(self, F, x, *args, **kwargs):
        c5 = self.conv1x1_p5(x)
        c4 = self.conv1x1_p4(args[0])
        c3 = self.conv1x1_p3(args[1])
        c2 = self.conv1x1_p2(args[2])

        c4_plus = c5 + c4
        c3_plus = c4_plus + c3
        c2_plus = c3_plus + c2

        p5 = self.head5(self.conv3x3_p5(c5))
        p4 = self.head4(self.conv3x3_p4(c4_plus))
        p3 = self.head3(self.conv3x3_p3(c3_plus))
        p2 = self.head2(self.conv3x3_p2(c2_plus))

        outputs = []
        out = self.head(F.concat(p5, p4, p3, p2, dim=1))
        outputs.append(out)

        if self.aux:
            aux_outs = self.aux_head(x, *args)
            outputs = outputs + aux_outs
        return tuple(outputs)


class _SegHead(nn.HybridBlock):
    def __init__(self, nclass, norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_SegHead, self).__init__()
        with self.name_scope():
            self.conv1 = ConvBlock(256, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv2 = nn.Conv2D(nclass, 1, in_channels=256)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
