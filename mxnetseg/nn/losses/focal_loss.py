# coding=utf-8

from mxnet.gluon.loss import Loss


class FocalLoss(Loss):
    """
    Focal loss for semantic segmentation.
    Reference: Lin, Tsung-Yi, et al. "Focal loss for dense object detection."
        Proceedings of the IEEE international conference on computer vision. 2017.
    """

    def __init__(self, gamma=2, batch_axis=0, ignore_label=-1, **kwargs):
        super(FocalLoss, self).__init__(None, batch_axis, **kwargs)
        self._gamma = gamma
        self._ignore_label = ignore_label

    def hybrid_forward(self, F, pred, label):
        log_pt = F.log_softmax(pred, axis=1)
        pt = F.exp(log_pt)
        loss = -F.pick(F.power(1 - pt, self._gamma) * log_pt, label, axis=1, keepdims=True)
        loss = F.where(label.expand_dims(axis=1) == self._ignore_label,
                       F.zeros_like(loss), loss)
        return F.mean(loss, axis=self._batch_axis, exclude=True)
