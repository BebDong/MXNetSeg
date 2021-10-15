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
    """
    Models:
        'DeepLabv3', 'DeepLabv3PlusX', 'DeepLabv3PlusR', 'SwiftResNet', 'SwiftResNetPr',
        'ACFNet', 'AlignSeg', 'AttentionToScale', 'AttaNet', 'BiSeNetX', 'BiSeNetR', 'DANet',
        'DenseASPP', 'FaPN', 'FCNResNet', 'FCNMobileNet', 'LadderDenseNet', 'PSPNet', 'SeENet',
        'SemanticFPN', 'SETR',
        'EPRNet', 'CANetv1', 'CANetv2'
    Datasets:
        'MSCOCO','ADE20K', 'Aeroscapes', 'BDD100K', 'CamVid', 'CityCoarse', 'Cityscapes', 'COCOStuff',
        'GATECH', 'Mapillary', 'MHPV1', 'NYUv2', 'PascalContext', 'SBD', 'SiftFlow',
        'StanfordBackground', 'SUNRGBD', 'PascalVOC', 'PascalVOCAug', 'WeizmannHorses'

    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--purpose', type=str, default='score',
                        help='score/speed')

    parser.add_argument('--model', type=str, default='PSPNet',
                        help='model name')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='backbone network')
    parser.add_argument('--dilate', action='store_true', default=True,
                        help='dilated backbone')
    parser.add_argument('--aux', action='store_true', default=True,
                        help='auxiliary segmentation head')
    parser.add_argument('--checkpoint', type=str,
                        default='pspnet_resnet50_cityscapes_20211013_203221.params',
                        help='checkpoint')

    parser.add_argument('--ctx', type=int, nargs='+', default=(0,),
                        help="ids of GPUs or leave None to use CPU")

    parser.add_argument('--data', type=str, default='Cityscapes',
                        help='dataset name')
    parser.add_argument('--crop', type=int, default=768,
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
                     hybridize=True)


if __name__ == '__main__':
    main()
