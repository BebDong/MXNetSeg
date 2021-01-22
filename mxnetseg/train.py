# coding=utf-8

import os
import sys
import yaml
import wandb
import argparse
from tqdm import tqdm

from mxnet import nd, autograd
from mxnet.log import get_logger
from gluoncv.utils import split_and_load
from gluoncv.utils.metrics import SegmentationMetric

# runtime environment
env_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if env_dir not in sys.path:
    sys.path.insert(0, env_dir)

from mxnetseg.engine import FitFactory
from mxnetseg.data import get_dataset_info
from mxnetseg.tools import root_dir, get_contexts, get_strftime, cudnn_auto_tune, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="MXNet segmentation",
        epilog="python train.py --model fcn --ctx 0 1 2 3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='fcn',
                        help='model name')
    parser.add_argument('--ctx', type=int, nargs='+', default=(0, 1),
                        help='GPU id or leave None to use CPU')
    parser.add_argument('--wandb', type=str, default='wandb-demo',
                        help='project name of wandb')
    parser.add_argument('--log-interval', type=int, default=5,
                        help='batch interval for log')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    # auto config when using wandb sweep, consistent with sweep.yaml
    parser.add_argument('--lr', type=float, default=None,
                        help='init learning rate')
    parser.add_argument('--wd', type=float, default=None,
                        help='weight decay')

    arg = parser.parse_args()
    return arg


def fit(ctx, log_interval=5, no_val=False, logger=None):
    net = FitFactory.get_model(wandb.config, ctx)
    train_iter, num_train = FitFactory.data_iter(wandb.config.data_name, wandb.config.bs_train,
                                                 root=get_dataset_info(wandb.config.data_name)[0],
                                                 split='train',  # sometimes would be 'trainval'
                                                 mode='train',
                                                 base_size=wandb.config.base_size,
                                                 crop_size=wandb.config.crop_size)
    val_iter, num_valid = FitFactory.data_iter(wandb.config.data_name, wandb.config.bs_val,
                                               shuffle=False, last_batch='keep',
                                               root=get_dataset_info(wandb.config.data_name)[0],
                                               split='val',
                                               base_size=wandb.config.base_size,
                                               crop_size=wandb.config.crop_size)
    criterion = FitFactory.get_criterion(wandb.config.aux, wandb.config.aux_weight,
                                         # focal_kwargs={'alpha': 1.0, 'gamma': 0.5},
                                         # sensitive_kwargs={
                                         #     'nclass': get_dataset_info(wandb.config.data_name)[1],
                                         #     'alpha': 1.0,
                                         #     'gamma': 1.0}
                                         )
    trainer = FitFactory.create_trainer(net, wandb.config, iters_per_epoch=len(train_iter))
    metric = SegmentationMetric(nclass=get_dataset_info(wandb.config.data_name)[1])

    wandb.config.num_train = num_train
    wandb.config.num_valid = num_valid

    t_start = get_strftime()
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
            loss_temp = .0  # sum up all sample loss
            for loss in loss_gpus:
                loss_temp += loss.sum().asscalar()
            train_loss += (loss_temp / wandb.config.bs_train)
            tbar.set_description('Epoch %d, training loss %.5f' % (epoch, train_loss / (i + 1)))
            if (i % log_interval == 0) or (i + 1 == len(train_iter)):
                wandb.log({f'train_loss_batch, interval={log_interval}': train_loss / (i + 1)})
        wandb.log({'train_loss_epoch': train_loss / (len(train_iter) + 1),
                   'custom_step': epoch})

        if not no_val:
            cudnn_auto_tune(False)
            val_loss = .0
            vbar = tqdm(val_iter)
            for i, (data, target) in enumerate(vbar):
                gpu_datas = split_and_load(data=data, ctx_list=ctx, even_split=False)
                gpu_targets = split_and_load(data=target, ctx_list=ctx, even_split=False)
                loss_temp = .0
                for gpu_data, gpu_target in zip(gpu_datas, gpu_targets):
                    loss_gpu = criterion(*net(gpu_data), gpu_target)
                    loss_temp += loss_gpu.sum().asscalar()
                    metric.update(gpu_target, net.evaluate(gpu_data))
                vbar.set_description('Epoch %d, val PA %.4f, mIoU %.4f'
                                     % (epoch, metric.get()[0], metric.get()[1]))
                val_loss += (loss_temp / wandb.config.bs_val)
                nd.waitall()
            pix_acc, mean_iou = metric.get()
            wandb.log({'val_PA': pix_acc, 'val_mIoU': mean_iou,
                       'val_loss': val_loss / len(val_iter) + 1,
                       'custom_step': epoch})
            metric.reset()
            if mean_iou > best_score:
                save_checkpoint(model=net,
                                model_name=wandb.config.model_name.lower(),
                                backbone=wandb.config.backbone.lower(),
                                data_name=wandb.config.data_name.lower(),
                                time_stamp=wandb.config.start_time,
                                is_best=True)
                best_score = mean_iou
            cudnn_auto_tune(True)

    save_checkpoint(model=net,
                    model_name=wandb.config.model_name.lower(),
                    backbone=wandb.config.backbone.lower(),
                    data_name=wandb.config.data_name.lower(),
                    time_stamp=wandb.config.start_time,
                    is_best=False)


def main():
    args = parse_args()

    with open('config.yml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        if not args.model == cfg.get('model_name'):
            raise RuntimeError(f"Inconsistent model name: "
                               f"{args.model} v.s. {cfg.get('model_name')}")

    wandb.init(job_type='train',
               dir=root_dir(),
               project=args.wandb,
               config=cfg)

    # if using wandb sweep
    if args.lr and args.wd:
        wandb.config.lr = args.lr
        wandb.config.wd = args.wd

    ctx = get_contexts(args.ctx)
    wandb.config.ctx = ctx  # add contexts to config

    logger = get_logger(name='train', level=10)

    fit(ctx=ctx,
        log_interval=args.log_interval,
        no_val=args.no_val,
        logger=logger)


if __name__ == '__main__':
    main()
