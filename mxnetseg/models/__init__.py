# coding=utf-8

from mxnet import cpu

from .base import *

from .deeplab import *
from .swiftnet import *
from .acfnet import ACFNet
from .att2scale import AttentionToScale
from .attanet import AttaNet
from .bisenet import *
from .danet import DANet
from .denseaspp import DenseASPP
from .eprnet import EPRNet
from .fcn import FCNResNet, FCNMobileNet
from .ladderdensenet import LadderDenseNet
from .pspnet import PSPNet
from .seenet import SeENet
from .semanticfpn import SemanticFPN
from .setr import SETR
from .canet import *

from mxnetseg.tools import validate_checkpoint, MODELS


class ModelFactory:
    def __init__(self, name):
        self._name = name
        self._class = MODELS[name]

    def get_model(self, model_kwargs, resume=None, lr_mult=1, backbone_init_manner='cls',
                  backbone_ckpt=None, ctx=(cpu())):
        net = self._class(**model_kwargs)
        if resume:
            checkpoint = validate_checkpoint(self._name, resume)
            net.load_parameters(checkpoint)
            print(f"Checkpoint loaded: {checkpoint}")
        elif backbone_init_manner is None:
            net.initialize()
            print("Random initialized.")
        elif backbone_init_manner == 'cls':
            net.head.initialize()
            net.head.collect_params().setattr('lr_mult', lr_mult)
            if net.aux:
                net.aux_head.initialize()
                net.aux_head.collect_params().setattr('lr_mult', lr_mult)
            print("Pretrained backbone on ImageNet loaded & head random initialized.")
        elif backbone_init_manner == 'seg':
            checkpoint = validate_checkpoint(self._name, backbone_ckpt)
            net.load_parameters(checkpoint, allow_missing=True, ignore_extra=True)
            print(f"Pretrained checkpoint loaded: {checkpoint}")
        else:
            raise RuntimeError(f"Unknown backbone init manner: {backbone_init_manner}")

        net.collect_params().reset_ctx(ctx=ctx)
        return net
