# coding=utf-8

import os
import time
import platform
import numpy as np
from tqdm import tqdm
from mxnet import nd, autograd
from mxnet.log import get_logger
from mxnet.gluon.data import DataLoader
from gluoncv.utils.metrics import SegmentationMetric
from gluoncv.model_zoo.segbase import MultiEvalModel

from mxnetseg.data import DataFactory
from mxnetseg.models import ModelFactory
import mxnetseg.tools as my_tools

logger = get_logger(name='eval', level=20)


class EvalHelper:
    def __init__(self, args):
        self._args = args
        self._ctx = my_tools.get_contexts(args.ctx)
        self._model_factory = ModelFactory(args.model)
        self._data_factory = DataFactory(args.data)

        self.net = self._get_model()
        self.data_set = self._data_set()
        self.scales = self._scales()

    def _get_model(self):
        norm_layer, norm_kwargs = my_tools.build_norm_layer('bn')
        model_kwargs = {
            'nclass': self._data_factory.num_class,
            'backbone': self._args.backbone,
            'aux': self._args.aux,
            'base_size': self._args.base,
            'crop_size': self._args.crop,
            'norm_layer': norm_layer,
            'norm_kwargs': norm_kwargs,
            'dilate': self._args.dilate,
            'pretrained_base': False,
        }
        net = self._model_factory.get_model(model_kwargs, self._args.checkpoint, ctx=self._ctx)
        return net

    def _data_set(self):
        transform = my_tools.image_transform()
        if self._args.mode == 'test':
            dataset = self._data_factory.seg_dataset(split='test', mode='test', transform=transform)
        elif self._args.mode == 'testval':
            dataset = self._data_factory.seg_dataset(split='val', mode='test', transform=transform)
        elif self._args.mode == 'val':
            dataset = self._data_factory.seg_dataset(split='val', mode='testval', transform=transform)
        else:
            raise RuntimeError(f"Unknown mode: {self._args.mode}")
        return dataset

    def _scales(self):
        if self._args.ms:
            if 'city' in self._args.data.lower():
                scales = (0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25)
            elif self._args.data.lower() == 'gatech':
                scales = (0.5, 0.8, 1.0, 1.2, 1.4)
            else:
                scales = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
        else:
            scales = (1.0,)
        return scales

    @staticmethod
    def _sample(shape, ctx) -> nd.NDArray:
        if isinstance(shape, (list, tuple)):
            h = shape[0]
            w = shape[1]
        else:
            h = shape
            w = shape
        sample = nd.random.uniform(shape=(1, 3, h, w), ctx=ctx)
        return sample

    def speed(self, data_size=(1024, 1024), iterations=1000, warm_up=500, hybridize=True):
        if hybridize:
            self.net.hybridize(static_alloc=True)
        sample = self._sample(data_size, self._ctx[0])

        logger.info(f'Warm-up starts for {warm_up} forward passes...')
        for _ in range(warm_up):
            with autograd.record(False):
                self.net.evaluate(sample)
        nd.waitall()

        logger.info(f'Evaluate inference speed for {iterations} forward passes...')
        start = time.time()
        for _ in range(iterations):
            with autograd.record(False):
                self.net.evaluate(sample)
        nd.waitall()
        time_cost = time.time() - start

        logger.info('Total time: %.2fs, latency: %.2fms, FPS: %.1f'
                    % (time_cost, time_cost / iterations * 1000, iterations / time_cost))

    @staticmethod
    def _prediction_dir(save_dir, data_name):
        if platform.system() == 'Linux':
            save_dir = os.path.join(my_tools.root_dir(), 'color_results')
        save_dir = os.path.join(save_dir, data_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        return save_dir

    @staticmethod
    def _mask(output, data_name, test_citys=False):  # set True when submit to Cityscapes
        predict = nd.squeeze(nd.argmax(output, 1)).asnumpy()
        if test_citys:
            mask = my_tools.city_train2label(predict)
        else:
            mask = my_tools.my_color_palette(predict, data_name)
        return mask

    def _colored_predictions(self, evaluator, dataset, data_name, save_dir):
        save_dir = self._prediction_dir(save_dir, data_name)
        bar = tqdm(dataset)
        for _, (img, dst) in enumerate(bar):
            img = img.expand_dims(0)
            output = evaluator.parallel_forward(img)[0]
            mask = self._mask(output, data_name)
            save_name = dst.split('.')[0] + '.png'
            save_path = os.path.join(save_dir, save_name)
            dir_path, _ = os.path.split(save_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            mask.save(save_path)

    @staticmethod
    def _eval_scores(evaluator, dataset, nclass):
        metric = SegmentationMetric(nclass)
        val_iter = DataLoader(dataset, batch_size=1, last_batch='keep')
        bar = tqdm(val_iter)
        for _, (img, label) in enumerate(bar):
            pred = evaluator.parallel_forward(img)[0]
            metric.update(label, pred)
            bar.set_description("PA: %.4f    mIoU: %.4f" % metric.get())
        pix_acc, miou = metric.get()
        total_inter = metric.total_inter
        total_union = metric.total_union
        per_class_iou = 1.0 * total_inter / (np.spacing(1) + total_union)
        return pix_acc, miou, per_class_iou

    def eval(self):
        self.net.hybridize()
        evaluator = MultiEvalModel(module=self.net,
                                   nclass=self._data_factory.num_class,
                                   ctx_list=self._ctx,
                                   flip=self._args.ms,
                                   scales=self.scales)
        logger.info(f'With scales: {self.scales}')

        if 'test' in self._args.mode:
            self._colored_predictions(evaluator, self.data_set, self._args.data.lower(),
                                      save_dir=self._args.save_dir)
        else:
            pa, miou, per_class_iou = self._eval_scores(evaluator, self.data_set,
                                                        nclass=self._data_factory.num_class)
            for i, score in enumerate(per_class_iou):
                logger.info('class {0:2} ==> IoU {1:2}'.format(i, round(score * 100, 2)))
            logger.info("PA: %.2f, mIoU: %.2f" % (pa * 100, miou * 100))
