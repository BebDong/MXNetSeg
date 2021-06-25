# coding=utf-8

import os
import sys
import yaml
import argparse
from gluoncv.utils import check_version

env_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if env_dir not in sys.path:
    sys.path.insert(0, env_dir)
check_version('0.8.0')

from mxnetseg.engine import train


def parse_args():
    parser = argparse.ArgumentParser(
        description="Model training",
        epilog="python train.py --ctx 0 1 2 3 --wandb demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--ctx', type=int, nargs='+', default=(0,),
                        help='GPU id or leave None to use CPU')
    parser.add_argument('--wandb', type=str, default='debug',
                        help='project name of wandb')
    parser.add_argument('--log-interval', type=int, default=5,
                        help='batch interval for logging')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    # auto config when using wandb sweep, consistent with sweep.yaml
    parser.add_argument('--lr', type=float, default=None,
                        help='init learning rate')
    parser.add_argument('--wd', type=float, default=None,
                        help='weight decay')

    return parser.parse_args()


def main():
    args = parse_args()
    with open('config.yml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    train(cfg,
          ctx_lst=args.ctx,
          project_name=args.wandb,
          log_interval=args.log_interval,
          no_val=args.no_val,
          lr=args.lr,
          wd=args.wd)


if __name__ == '__main__':
    main()
