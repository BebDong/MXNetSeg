# coding=utf-8

import os
import time
import platform
import numpy as np

import mxnet as mx
from mxnet import nd
from mxnet.gluon.nn import BatchNorm
from mxnet.gluon.contrib.nn.basic_layers import SyncBatchNorm

from gluoncv.nn import GroupNorm

__all__ = ['cudnn_auto_tune', 'get_contexts', 'get_strftime', 'build_norm_layer',
           'list_to_str', 'misclassified_pixels', 'misclassified_prop']


def cudnn_auto_tune(tune: bool = True):
    """Linux/Windows: MXNet cudnn auto-tune"""
    tag = 1 if tune else 0
    if platform.system() == 'Linux':
        os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = str(tag)
    else:
        os.system(f"set MXNET_CUDNN_AUTOTUNE_DEFAULT={tag}")


def get_contexts(ctx_id):
    """
    Return all available GPUs, or [mx.cpu()] if there is no GPU

    :param ctx_id: list of context ids
    :return: list of MXNet contexts
    """

    ctx_list = []
    if ctx_id:
        try:
            for i in ctx_id:
                ctx = mx.gpu(i)
                _ = nd.array([0], ctx=ctx)
                ctx_list.append(ctx)
        except mx.base.MXNetError:
            raise RuntimeError(f"=> unknown gpu id: {ctx_id}")
    else:
        ctx_list = [mx.cpu()]
    return ctx_list


def build_norm_layer(bn: str, num_ctx=1):
    """get corresponding norm layer"""
    if bn == 'bn':
        norm_layer = BatchNorm
        norm_kwargs = None
    elif bn == 'sbn':
        norm_layer = SyncBatchNorm
        norm_kwargs = {'num_devices': num_ctx}
    elif bn == 'gn':
        norm_layer = GroupNorm
        norm_kwargs = {'ngroups': 32}
    else:
        raise NotImplementedError(f"Unknown batch normalization layer: {bn}")
    return norm_layer, norm_kwargs


def get_strftime(str_format='%Y%m%d_%H%M%S'):
    """string format time"""
    return time.strftime(str_format, time.localtime())


def list_to_str(lst):
    """convert list/tuple to string split by space"""
    result = " ".join(str(i) for i in lst)
    return result


def misclassified_pixels(prob: nd.NDArray, label: nd.NDArray, ignore_label: int = -1) -> np.ndarray:
    """
    return misclassified pixels.
    :param prob: the predicted probability with shape CHW
    :param label: the ground truth label with shape HW
    :param ignore_label: ignored label
    :return: numpy array of shape HW where 0 indicates misclassified pixels
    """
    # needs to process on cpu
    prob = prob.as_in_context(mx.cpu())
    label = label.as_in_context(mx.cpu())

    # determine equal or not to get misclassified pixels
    pred = nd.squeeze(nd.argmax(prob, axis=0)).astype('int32')
    mis_classify = (pred == label).asnumpy()

    # deal with ignored label via numpy
    label = label.asnumpy()
    mis_classify[label == ignore_label] = 1

    return mis_classify


def misclassified_prop(mis_pixels: np.ndarray, prob: nd.NDArray) -> np.ndarray:
    """
    get the probability distribution of misclassified pixels.
    :param mis_pixels: misclassified pixels of shape HW where 0 indicates misclassified pixels
    :param prob: the predicted probability with shape CHW
    :return: numpy array of shape NC where N is the total number of misclassified pixels
    """
    prob = prob.as_in_context(mx.cpu())
    prob = nd.transpose(nd.reshape(prob, shape=(0, -1)), axes=(1, 0))
    flag_bit = np.reshape(mis_pixels, newshape=(-1))
    mis_index = nd.array(np.where(flag_bit == 0)).squeeze()
    return prob[mis_index, :].asnumpy()
