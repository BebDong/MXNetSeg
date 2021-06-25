# coding=utf-8

from mxnet import nd
from mxnet.gluon.loss import Loss

__all__ = ['MixedCELoss', 'WeightedCELoss', 'BootstrappedCELoss']


class MixedCELoss(Loss):
    """
    Cross-entropy loss with multiple auxiliary losses
    """

    def __init__(self, aux=False, aux_weight=None, batch_axis=0, ignore_label=-1, **kwargs):
        super(MixedCELoss, self).__init__(None, batch_axis, **kwargs)
        self.aux = aux
        if aux and aux_weight:
            self.aux_weight = (aux_weight,) if isinstance(aux_weight, float) else aux_weight
        self._ignore_label = ignore_label

    def _base_forward(self, F, pred, label):
        log_pt = F.log_softmax(pred, axis=1)
        loss = -F.pick(log_pt, label, axis=1, keepdims=True)
        loss = F.where(label.expand_dims(axis=1) == self._ignore_label,
                       F.zeros_like(loss), loss)
        return F.mean(loss, axis=self._batch_axis, exclude=True)

    def _aux_forward(self, F, *inputs, **kwargs):
        assert len(self.aux_weight) == (len(inputs) - 2)  # no weights for pred1 and label
        label = inputs[-1]
        loss = self._base_forward(F, inputs[0], label)
        for i in range(len(self.aux_weight)):
            aux_loss = self._base_forward(F, inputs[i + 1], label)
            loss = loss + aux_loss * self.aux_weight[i]
        return loss

    def hybrid_forward(self, F, *inputs, **kwargs):
        if self.aux:
            return self._aux_forward(F, *inputs, **kwargs)
        else:
            return self._base_forward(F, *inputs)


class WeightedCELoss(Loss):
    """
    Weighted cross-entropy loss.
    """

    def __init__(self, weight, batch_axis=0, ignore_label=-1, **kwargs):
        super(WeightedCELoss, self).__init__(None, batch_axis, **kwargs)
        self._weight = self.params.get_constant('_weight', nd.array(weight, dtype='float32'))
        self._ignore_label = ignore_label

    def hybrid_forward(self, F, pred, label, _weight):
        log_pt = F.log_softmax(pred, axis=1)
        loss = -F.pick(log_pt, label, axis=1, keepdims=True)
        gather_weight = F.gather_nd(_weight, F.expand_dims(label, axis=0))
        loss = F.multiply(loss, F.expand_dims(gather_weight, axis=1))
        loss = F.where(label.expand_dims(axis=1) == self._ignore_label,
                       F.zeros_like(loss), loss)
        return F.mean(loss, axis=self._batch_axis, exclude=True)


class BootstrappedCELoss(Loss):
    """
    Online bootstrapping of hard training pixels.
    Consider image crops within a batch as one crop.
    Reference:
        [1] Z. Wu, C. Shen, and A. van den Hengel, “Bridging Category-level and Instance-level
        Semantic Image Segmentation,” ArXiv, 2016.
    """

    def __init__(self, min_k, loss_th, batch_axis=0, ignore_label=-1, **kwargs):
        super(BootstrappedCELoss, self).__init__(None, batch_axis, **kwargs)
        self._K = min_k
        self._eps = 1e-5
        self._thresh = self.params.get_constant('_thresh', nd.array([loss_th, ], dtype='float32'))
        self._ignore_label = ignore_label

    def _greater_loss(self, F, sorted_loss, threshold):
        loss = F.where(sorted_loss > threshold, sorted_loss, F.zeros_like(sorted_loss))
        loss = F.sum(loss) / (F.sum(loss > .0) + self._eps)
        return loss

    def hybrid_forward(self, F, pred, label, _thresh):
        log_pt = F.log_softmax(pred, axis=1)
        loss = -F.pick(log_pt, label, axis=1, keepdims=True)
        loss = F.where(label.expand_dims(axis=1) == self._ignore_label,
                       F.zeros_like(loss), loss)
        sorted_loss = F.sort(F.reshape(loss, shape=(-1,)), axis=0, is_ascend=False)
        return F.where(
            F.greater(sorted_loss[self._K], _thresh),
            self._greater_loss(F, sorted_loss, _thresh),
            F.mean(F.slice_axis(sorted_loss, axis=0, begin=0, end=self._K))
        )
