# coding=utf-8

import mxnet as mx

from .base import *
from .att2scale import *
from .bisenet import *
from .deeplab import *
from .denseaspp import *
from .fcn import *
from .ladder import *
from .pspnet import *
from .seenet import *
from .swiftnet import *
from .danet import *
from .acfnet import *
from .sfpn import *

from .canet import *

from mxnetseg.tools import validate_checkpoint

_seg_models = {
    'fcn': get_fcn,
    'att2scale': get_att2scale,
    'bisenet': get_bisenet,
    'deeplabv3': get_deeplabv3,
    'deeplabv3plus': get_deeplabv3plus,
    'denseaspp': get_denseaspp,
    'ladder': get_ladder,
    'pspnet': get_pspnet,
    'seenet': get_seenet,
    'swiftnet': get_swiftnet,
    'swiftnetpr': get_swiftnet_pyramid,
    'danet': get_danet,
    'acfnet': get_acfnet,
    'semanticfpn': get_semantic_fpn,

    'canet': get_canet,
}


def get_model_by_name(name, model_kwargs, resume=None, lr_mult=1, ctx=(mx.cpu())):
    """get initialized model by name"""
    get_model = _seg_models[name.lower()]
    model = get_model(**model_kwargs)
    if resume:
        checkpoint = validate_checkpoint(name, resume)
        model.load_parameters(checkpoint)
    else:
        _init_model(model, model_kwargs['pretrained_base'], lr_mult)
    model.collect_params().reset_ctx(ctx=ctx)
    return model


def _init_model(model, pretrained_base, lr_mult=1):
    """init model params"""
    if not pretrained_base:
        model.initialize()
    else:
        model.head.initialize()
        model.head.collect_params().setattr('lr_mult', lr_mult)
        if model.aux:
            model.auxlayer.initialize()
            model.auxlayer.collect_params().setattr('lr_mult', lr_mult)
