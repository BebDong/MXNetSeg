# coding=utf-8

import os
import sys
import argparse

# runtime environment
env_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if env_dir not in sys.path:
    sys.path.insert(0, env_dir)

from mxnetseg.engine import EvalHelper


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--purpose', type=str, default='score',
                        help='score/speed')

    parser.add_argument('--model', type=str, default='CANetv2',
                        help='model name')
    parser.add_argument('--backbone', type=str, default='resnet101',
                        help='backbone network')
    parser.add_argument('--dilate', action='store_true', default=True,
                        help='dilated backbone')
    parser.add_argument('--aux', action='store_true', default=False,
                        help='auxiliary segmentation head')
    parser.add_argument('--checkpoint', type=str,
                        default='net.params',
                        help='checkpoint')

    parser.add_argument('--ctx', type=int, nargs='+', default=(0,),
                        help="ids of GPUs or leave None to use CPU")

    parser.add_argument('--data', type=str, default='Cityscapes',
                        help='dataset name')
    parser.add_argument('--crop', type=int, default=480,
                        help='crop size')
    parser.add_argument('--base', type=int, default=2048,
                        help='random scale base size')
    parser.add_argument('--mode', type=str, default='val',
                        choices=('val', 'testval', 'test'),
                        help='evaluation/prediction on val/test set')
    parser.add_argument('--ms', action='store_true', default=False,
                        help='enable multi-scale and flip skills')
    parser.add_argument('--save-dir', type=str,
                        default='C:\\Users\\BedDong\\Desktop',
                        help='path to save predictions for windows')
    return parser.parse_args()


def main():
    args = parse_args()

    helper = EvalHelper(args)
    if args.purpose == 'score':
        helper.eval()
    else:
        helper.speed(data_size=(480, 480),
                     iterations=1000,
                     warm_up=500,
                     hybridize=False)


if __name__ == '__main__':
    main()
