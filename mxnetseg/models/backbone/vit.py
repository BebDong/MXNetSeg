# coding=utf-8
# TODO: 2D interpolation of the pre-trained position embeddings when fine-tuning on higher resolution

from mxnet import init
from mxnet.gluon import nn
from mxnetseg.nn import Activation
from mxnetseg.tools import validate_checkpoint

__all__ = ['VisionTransformer', 'get_vit', 'vit_base_16', 'vit_large_16', 'vit_huge_16']


class VisionTransformer(nn.HybridBlock):
    """
    Vision Transformer

    Reference:
        X. Zhai et al., “An Image is Worth 16x16 Words: Transformers for Image Recognition
        at Scale,” in International Conference of Learning Representation, 2021.
    """

    def __init__(self, patch_size, depth, embed_dim, hidden_size, heads, img_size, classes,
                 embed_drop=0., qkv_bias=True, att_drop=0., drop=0.1, activation='gelu',
                 layer_norm_eps=1e-6, pool='cls', **kwargs):
        super(VisionTransformer, self).__init__()
        assert pool in ('cls', 'mean'), 'pool type must be either cls (cls token) or ' \
                                        'mean (mean pooling)'
        self.pool = pool
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        with self.name_scope():
            self.patch_embed = _PatchEmbedding(img_size, patch_size, embed_dim)
            self.cls_token = self.params.get('cls_token', shape=(1, 1, embed_dim), init=init.Zero())
            self.pos_embed = self.params.get('pos_embed',
                                             shape=(1, self.patch_embed.num_patches + 1, embed_dim),
                                             init=init.Zero())
            self.embed_dropout = nn.Dropout(embed_drop) if embed_drop else None

            self.blocks = nn.HybridSequential()
            for i in range(depth):
                self.blocks.add(
                    _TransformerEncoder(embed_dim, heads, hidden_size, qkv_bias, att_drop,
                                        drop, activation, layer_norm_eps=layer_norm_eps)
                )
            self.head = nn.HybridSequential()
            self.head.add(
                nn.LayerNorm(epsilon=layer_norm_eps, in_channels=embed_dim),
                nn.Dense(classes)
            )

    def hybrid_forward(self, F, x, *args, **kwargs):
        B = x.shape[0]  # need to get batch_size, therefore not support hybridize() function
        cls_token = kwargs['cls_token']
        cls_token = F.broadcast_to(cls_token, shape=(B, 0, 0))
        pos_embed = kwargs['pos_embed']

        x = self.patch_embed(x)
        x = F.concat(x, cls_token, dim=1)
        x = F.broadcast_add(x, pos_embed)
        if self.embed_dropout:
            x = self.embed_dropout(x)

        x = self.blocks(x)
        x = F.mean(x, axis=1) if self.pool == 'mean' else x[:, 0, :]
        x = self.head(x)

        return x


class _PatchEmbedding(nn.HybridBlock):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768):
        super(_PatchEmbedding, self).__init__()
        self._num_patches = (img_size // patch_size) ** 2
        self.linear = nn.Conv2D(embed_dim, patch_size, patch_size)

    @property
    def num_patches(self):
        return self._num_patches

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.linear(x)  # BC(num_patches)(num_patches)
        x = F.transpose(F.reshape(x, shape=(0, 0, -1)), axes=(0, 2, 1))  # BNC
        return x


class _TransformerEncoder(nn.HybridBlock):
    def __init__(self, units, heads, hidden_size, qkv_bias=False, att_drop=0., drop=0.,
                 activation='gelu', layer_norm_eps=1e-12):
        super(_TransformerEncoder, self).__init__()
        with self.name_scope():
            self.norm1 = nn.LayerNorm(epsilon=layer_norm_eps, in_channels=units)
            self.att = _MultiHeadAttention(units, heads, qkv_bias, att_drop, drop)
            self.norm2 = nn.LayerNorm(epsilon=layer_norm_eps, )
            self.mlp = _MLP(units, hidden_size, activation, drop)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = x + self.att(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _MultiHeadAttention(nn.HybridBlock):
    def __init__(self, units, heads, qkv_bias=False, att_drop=0., proj_drop=0.):
        super(_MultiHeadAttention, self).__init__()
        head_units = units // heads
        self.heads = heads
        self.scale = head_units ** 0.5
        with self.name_scope():
            self.att_qkv = nn.Dense(3 * units, use_bias=qkv_bias, flatten=False)
            self.att_dropout = nn.Dropout(att_drop) if att_drop else None
            self.proj = nn.Dense(units, flatten=False)
            self.proj_dropout = nn.Dropout(proj_drop) if proj_drop else None

    def hybrid_forward(self, F, x, *args, **kwargs):
        query, key, value = F.split(self.att_qkv(x), 3, axis=-1)  # BN(head_units * heads)
        query = F.transpose(F.reshape(query, (0, 0, self.heads, -1)), axes=(0, 2, 1, 3))
        key = F.transpose(F.reshape(key, (0, 0, self.heads, -1)), axes=(0, 2, 1, 3))
        value = F.transpose(F.reshape(value, (0, 0, self.heads, -1)), axes=(0, 2, 1, 3))
        scores = F.batch_dot(query, key, transpose_b=True)
        scores = F.softmax(scores, temperature=self.scale)
        if self.att_dropout:
            scores = self.att_dropout(scores)
        out = F.transpose(F.batch_dot(scores, value), axes=(0, 2, 1, 3))
        out = F.reshape(out, shape=(0, 0, -1))
        out = self.proj(out)
        if self.proj_dropout:
            out = self.proj_dropout(out)
        return out


class _MLP(nn.HybridBlock):
    def __init__(self, units, hidden_size, activation='gelu', drop=0.):
        super(_MLP, self).__init__()
        self.dropout = drop
        with self.name_scope():
            self.fc1 = nn.Dense(hidden_size, flatten=False)
            self.act = Activation(activation)
            self.fc2 = nn.Dense(units, flatten=False)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.fc1(x)
        x = self.act(x)
        x = F.Dropout(x, p=self.dropout)
        x = self.fc2(x)
        x = F.Dropout(x, p=self.dropout)
        return x


def get_vit(patch_size, depth, embed_dim, hidden_size, heads, pretrained=False, ckpt_name=None,
            **kwargs):
    model = VisionTransformer(patch_size, depth, embed_dim, hidden_size, heads, **kwargs)
    if pretrained:
        model_weight = validate_checkpoint('VisionTransformer', ckpt_name)
        model.load_parameters(model_weight)
    return model


def vit_base_16(pretrained=False, **kwargs):
    return get_vit(16, 12, 768, 3072, 12, pretrained, ckpt_name='vit_base_patch16_224.params',
                   **kwargs)


def vit_large_16(pretrained=False, **kwargs):
    return get_vit(16, 24, 1024, 4096, 16, pretrained, ckpt_name='vit_large_patch16_224.params',
                   **kwargs)


def vit_huge_16(pretrained=False, **kwargs):
    return get_vit(16, 32, 1280, 5120, 16, pretrained, ckpt_name='vit_huge_patch16_224.params',
                   **kwargs)
