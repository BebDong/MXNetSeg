# coding=utf-8

from mxnet import cpu
from .base import *
from .deeplab import *
from .swiftnet import *
from .acfnet import ACFNet
from .alignseg import AlignSeg
from .att2scale import AttentionToScale
from .attanet import AttaNet
from .bisenet import *
from .danet import DANet
from .denseaspp import DenseASPP
from .eprnet import EPRNet
from .fapn import FaPN
from .fcn import FCNResNet, FCNMobileNet
from .ladderdensenet import LadderDenseNet
from .pspnet import PSPNet
from .seenet import SeENet
from .semanticfpn import SemanticFPN
from .setr import SETR
from .canet import *
from mxnetseg.utils import validate_checkpoint, MODELS


class ModelFactory:
    def __init__(self, name):
        self._name = name
        self._class = MODELS[name]

    def get_model(self, model_kwargs, resume=None, lr_mult=1, backbone_init_manner='cls',
                  backbone_ckpt=None, prior_classes=None, ctx=(cpu())):
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
            print("ImageNet pre-trained backbone loaded & head random initialized.")
        elif backbone_init_manner == 'seg':
            model_kwargs['nclass'] = prior_classes
            pretrain = self._class(**model_kwargs)
            checkpoint = validate_checkpoint(self._name, backbone_ckpt)
            pretrain.load_parameters(checkpoint)

            # only support resnet for now
            net.conv1 = pretrain.conv1
            net.bn1 = pretrain.bn1
            net.relu = pretrain.relu
            net.maxpool = pretrain.maxpool
            net.layer1 = pretrain.layer1
            net.layer2 = pretrain.layer2
            net.layer3 = pretrain.layer3
            net.layer4 = pretrain.layer4

            net.head.initialize()
            if net.aux:
                net.aux_head.initialize()

            print("Pre-trained segmentation model loaded & head random initialized.")
        else:
            raise RuntimeError(f"Unknown backbone init manner: {backbone_init_manner}")

        net.collect_params().reset_ctx(ctx=ctx)
        return net
