# coding=utf-8

import wandb
import platform
from tqdm import tqdm
from mxnet.log import get_logger
from mxnet import nd, autograd
from mxnet.gluon import Trainer
from mxnet.gluon.data import DataLoader
from gluoncv.utils import LRScheduler, split_and_load
from gluoncv.utils.metrics import SegmentationMetric

import mxnetseg.tools as my_tools
from mxnetseg.data import DataFactory
from mxnetseg.models import ModelFactory


def _get_criterion(aux, aux_weight, focal_kwargs=None, class_weight=None,
                   topk_kwargs=None, dice_kwargs=None):
    if focal_kwargs:
        from mxnetseg.nn import FocalLoss
        return FocalLoss(**focal_kwargs)
    if class_weight:
        from mxnetseg.nn import WeightedCELoss
        return WeightedCELoss(class_weight)
    if topk_kwargs:
        from mxnetseg.nn import BootstrappedCELoss
        return BootstrappedCELoss(**topk_kwargs)
    if dice_kwargs:
        from mxnetseg.nn import DiceLoss
        return DiceLoss(**dice_kwargs)
    else:
        from gluoncv.loss import MixSoftmaxCrossEntropyLoss as MixedCELoss
        # from mxnetseg.nn import MixedCELoss
        return MixedCELoss(aux, aux_weight=aux_weight)


def _lr_scheduler(mode, base_lr, target_lr, nepochs, iters_per_epoch,
                  step_epoch=None, step_factor=0.1, power=0.9):
    assert mode in ('constant', 'step', 'linear', 'poly', 'cosine')
    sched_kwargs = {
        'base_lr': base_lr,
        'target_lr': target_lr,
        'nepochs': nepochs,
        'iters_per_epoch': iters_per_epoch,
    }
    if mode == 'step':
        sched_kwargs['mode'] = 'step'
        sched_kwargs['step_epoch'] = step_epoch
        sched_kwargs['step_factor'] = step_factor
    elif mode == 'poly':
        sched_kwargs['mode'] = 'poly'
        sched_kwargs['power'] = power
    else:
        sched_kwargs['mode'] = mode
    return LRScheduler(**sched_kwargs)


def train(cfg, ctx_lst, project_name, log_interval=5, no_val=False, lr=None, wd=None):
    wandb.init(job_type='train', dir=my_tools.root_dir(), config=cfg, project=project_name)
    if lr and wd:
        wandb.config.lr = lr
        wandb.config.wd = wd

    ctx = my_tools.get_contexts(ctx_lst)
    wandb.config.ctx = ctx

    data_factory = DataFactory(wandb.config.data_name)
    model_factory = ModelFactory(wandb.config.model_name)

    norm_layer, norm_kwargs = my_tools.build_norm_layer(wandb.config.norm, len(ctx))
    model_kwargs = {
        'nclass': data_factory.num_class,
        'backbone': wandb.config.backbone,
        'pretrained_base': wandb.config.backbone_init.get('manner') == 'cls',
        'aux': wandb.config.aux,
        'crop_size': wandb.config.crop_size,
        'base_size': wandb.config.base_size,
        'dilate': wandb.config.dilate,
        'norm_layer': norm_layer,
        'norm_kwargs': norm_kwargs,
    }
    net = model_factory.get_model(model_kwargs,
                                  resume=wandb.config.resume,
                                  lr_mult=wandb.config.lr_mult,
                                  backbone_init_manner=wandb.config.backbone_init.get('manner'),
                                  backbone_ckpt=wandb.config.backbone_init.get('backbone_ckpt'),
                                  ctx=ctx)
    if net.symbolize:
        net.hybridize()

    num_worker = 0 if platform.system() == 'Windows' else 16
    train_set = data_factory.seg_dataset(split='train',  # sometimes would be 'trainval'
                                         mode='train', transform=my_tools.image_transform(),
                                         base_size=wandb.config.base_size,
                                         crop_size=wandb.config.crop_size)
    train_iter = DataLoader(train_set, wandb.config.bs_train, shuffle=True, last_batch='discard',
                            num_workers=num_worker)
    val_set = data_factory.seg_dataset(split='val', mode='val', transform=my_tools.image_transform(),
                                       base_size=wandb.config.base_size,
                                       crop_size=wandb.config.crop_size)
    val_iter = DataLoader(val_set, wandb.config.bs_val, shuffle=False, last_batch='keep',
                          num_workers=num_worker)
    wandb.config.num_train = len(train_set)
    wandb.config.num_valid = len(val_set)

    criterion = _get_criterion(wandb.config.aux, wandb.config.aux_weight)
    criterion.initialize(ctx=ctx)
    wandb.config.criterion = type(criterion)

    if wandb.config.optimizer == 'adam':
        trainer = Trainer(net.collect_params(), 'adam',
                          optimizer_params={'learning_rate': wandb.config.lr,
                                            'wd': wandb.config.wd,
                                            'beta1': wandb.config.adam.get('adam_beta1'),
                                            'beta2': wandb.config.adam.get('adam_beta2')})
    elif wandb.config.optimizer in ('sgd', 'nag'):
        scheduler = _lr_scheduler(mode=wandb.config.lr_scheduler,
                                  base_lr=wandb.config.lr,
                                  target_lr=wandb.config.target_lr,
                                  nepochs=wandb.config.epochs,
                                  iters_per_epoch=len(train_iter),
                                  step_epoch=wandb.config.step.get('step_epoch'),
                                  step_factor=wandb.config.step.get('step_factor'),
                                  power=wandb.config.poly.get('power'))
        trainer = Trainer(net.collect_params(), wandb.config.optimizer,
                          optimizer_params={'lr_scheduler': scheduler,
                                            'wd': wandb.config.wd,
                                            'momentum': wandb.config.momentum,
                                            'multi_precision': True})
    else:
        raise RuntimeError(f"Unknown optimizer: {wandb.config.optimizer}")

    metric = SegmentationMetric(data_factory.num_class)

    logger = get_logger(name='train', level=10)
    t_start = my_tools.get_strftime()
    logger.info(f'Training start: {t_start}')
    for k, v in wandb.config.items():
        logger.info(f'{k}: {v}')
    logger.info('-----> end hyper-parameters <-----')
    wandb.config.start_time = t_start

    best_score = .0
    for epoch in range(wandb.config.epochs):
        train_loss = .0
        tbar = tqdm(train_iter)
        for i, (data, target) in enumerate(tbar):
            gpu_datas = split_and_load(data, ctx_list=ctx)
            gpu_targets = split_and_load(target, ctx_list=ctx)
            with autograd.record():
                loss_gpus = [criterion(*net(gpu_data), gpu_target)
                             for gpu_data, gpu_target in zip(gpu_datas, gpu_targets)]
            for loss in loss_gpus:
                autograd.backward(loss)
            trainer.step(wandb.config.bs_train)
            nd.waitall()
            train_loss += sum([loss.mean().asscalar() for loss in loss_gpus]) / len(loss_gpus)
            tbar.set_description('Trn-epoch-%d, %s, loss %.5f'
                                 % (epoch, my_tools.get_strftime('%Y-%m-%d %H:%M:%S'),
                                    train_loss / (i + 1)))
            if (i % log_interval == 0) or (i + 1 == len(train_iter)):
                wandb.log({f'train_loss_batch, interval={log_interval}': train_loss / (i + 1)})

        wandb.log({'train_loss_epoch': train_loss / (len(train_iter)), 'custom_step': epoch})

        if not no_val:
            val_loss = .0
            vbar = tqdm(val_iter)
            for i, (data, target) in enumerate(vbar):
                gpu_datas = split_and_load(data=data, ctx_list=ctx, even_split=False)
                gpu_targets = split_and_load(data=target, ctx_list=ctx, even_split=False)
                loss_gpus = []
                for gpu_data, gpu_target in zip(gpu_datas, gpu_targets):
                    gpu_output = net(gpu_data)
                    loss_gpus.append(criterion(*gpu_output, gpu_target))
                    metric.update(gpu_target, gpu_output[0])
                val_loss += sum([loss.mean().asscalar() for loss in loss_gpus]) / len(loss_gpus)
                vbar.set_description('Val-epoch-%d, PA %.4f, mIoU %.4f'
                                     % (epoch, metric.get()[0], metric.get()[1]))
                nd.waitall()
            pix_acc, mean_iou = metric.get()
            wandb.log({'val_PA': pix_acc, 'val_mIoU': mean_iou,
                       'val_loss': val_loss / len(val_iter),
                       'custom_step': epoch})
            metric.reset()
            if mean_iou > best_score:
                my_tools.save_checkpoint(model=net,
                                         model_name=wandb.config.model_name.lower(),
                                         backbone=wandb.config.backbone.lower(),
                                         data_name=wandb.config.data_name.lower(),
                                         time_stamp=wandb.config.start_time,
                                         is_best=True)
                best_score = mean_iou

    my_tools.save_checkpoint(model=net,
                             model_name=wandb.config.model_name.lower(),
                             backbone=wandb.config.backbone.lower(),
                             data_name=wandb.config.data_name.lower(),
                             time_stamp=wandb.config.start_time,
                             is_best=False)
