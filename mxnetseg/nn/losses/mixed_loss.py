# coding=utf-8
# adapted from:
# https://github.com/PaddlePaddle/PaddleSeg/tree/release/v2.0/paddleseg/models/losses/mixed_loss.py

from mxnet.gluon.loss import Loss


class MixedLoss(Loss):
    """
    Weighted computations for multiple Loss with the same prediction and label.

    :arg losses: a list containing multiple loss classes
    :arg coef: weighting coefficients of multiple losses
    """

    def __init__(self, losses, coef, batch_axis=0, **kwargs):
        super(MixedLoss, self).__init__(None, batch_axis, **kwargs)
        if not isinstance(losses, (list, tuple)):
            raise TypeError(f'losses must be a list or tuple, but get {type(losses)}')
        if not isinstance(coef, (list, tuple)):
            raise TypeError(f'coef must be a list or tuple, but get {type(coef)}')
        assert len(losses) == len(coef), 'unequal length of losses and coef'
        self._losses = losses
        self._coef = coef

    def hybrid_forward(self, F, pred, label):
        losses = [loss(pred, label) * self._coef[i] for i, loss in enumerate(self._losses)]
        if len(losses) == 1:
            return losses[0]
        else:
            return F.add_n(*losses)
