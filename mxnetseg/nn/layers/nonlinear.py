# coding=utf-8

from mxnet.gluon import nn
from gluoncv.nn import ReLU6, HardSigmoid, HardSwish

__all__ = ['Mish', 'Activation']


class Mish(nn.HybridBlock):
    """
    Mish activation function.
    Reference:
        Misra D. Mish: A Self Regularized Non-Monotonic Neural Activation Function.
        arXiv preprint arXiv:1908.08681, 2019.
    """

    def __init__(self):
        super(Mish, self).__init__()

    def hybrid_forward(self, F, x, *args, **kwargs):
        return x * F.tanh(F.Activation(data=x, act_type='softrelu'))


class Activation(nn.HybridBlock):
    def __init__(self, act_func: str, **kwargs):
        super(Activation, self).__init__()
        with self.name_scope():
            if act_func in ('relu', 'sigmoid', 'softrelu', 'softsign', 'tanh'):
                self.act = nn.Activation(act_func)
            elif act_func == 'leaky':
                self.act = nn.LeakyReLU(**kwargs)
            elif act_func == 'prelu':
                self.act = nn.PReLU(**kwargs)
            elif act_func == 'selu':
                self.act = nn.SELU()
            elif act_func == 'elu':
                self.act = nn.ELU(**kwargs)
            elif act_func == 'gelu':
                self.act = nn.GELU()
            elif act_func == 'relu6':
                self.act = ReLU6()
            elif act_func == 'hard_sigmoid':
                self.act = HardSigmoid()
            elif act_func == 'swish':
                self.act = nn.Swish()
            elif act_func == 'hard_swish':
                self.act = HardSwish()
            elif act_func == 'mish':
                self.act = Mish()
            else:
                raise NotImplementedError(f"Not implemented activation: {act_func}")

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.act(x)
