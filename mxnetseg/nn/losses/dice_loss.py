# coding=utf-8

from mxnet.gluon.loss import Loss


class DiceLoss(Loss):
    def __init__(self, num_class, smooth=1, exponent=2, batch_axis=0, ignore_label=-1, **kwargs):
        super(DiceLoss, self).__init__(None, batch_axis, **kwargs)
        self._num_class = num_class
        self._smooth = smooth
        self._exponent = exponent
        self._ignore_label = ignore_label

    def hybrid_forward(self, F, pred, label):
        prob = F.softmax(pred, axis=1)

        valid_mask = F.expand_dims(label, axis=1) != self._ignore_label
        prob = F.multiply(prob, valid_mask.astype(prob.dtype))

        label = F.one_hot(label, depth=self._num_class, dtype=prob.dtype)
        label = F.transpose(label, axes=(0, 3, 1, 2))

        inter = F.sum(prob * label, axis=(1, 2, 3)) * 2
        den = F.sum(F.power(prob, self._exponent) + F.power(label, self._exponent), axis=(1, 2, 3))
        dice = (inter + self._smooth) / (den + self._smooth)
        return 1 - F.mean(dice, axis=self._batch_axis, exclude=True)
