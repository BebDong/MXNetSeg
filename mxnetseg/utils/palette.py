# coding=utf-8

import json
import numpy as np
from PIL import Image
from gluoncv.utils.viz import get_color_pallete
from .path import mapillary_config

__all__ = ['my_color_palette', 'city_train2label']

palette_name_mapping = {
    'pascalvoc': 'pascal_voc',
    'cityscapes': 'citys',
    'bdd100k': 'citys',  # BDD100k shares the same 19 semantic categories as Cityscapes
}


def city_train2label(npimg):
    """convert train ID to label ID for cityscapes"""
    valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                     23, 24, 25, 26, 27, 28, 31, 32, 33]
    pred = np.array(npimg, np.uint8)
    label = np.zeros(pred.shape)
    ids = np.unique(pred)
    for i in ids:
        label[np.where(pred == i)] = valid_classes[i]
    out_img = Image.fromarray(label.astype('uint8'))
    return out_img


def my_color_palette(npimg, dataset: str):
    """
    Visualize image and return PIL.Image with color palette.

    :param npimg: Single channel numpy image with shape `H, W, 1`
    :param dataset: dataset name
    """

    dataset = dataset.lower()
    if dataset in palette_name_mapping.keys():
        dataset = palette_name_mapping[dataset]

    if dataset in ('pascal_voc', 'ade20k', 'citys', 'mhpv1'):
        return get_color_pallete(npimg, dataset=dataset)
    elif dataset == 'camvid':
        npimg[npimg == -1] = 11
        out_img = Image.fromarray(npimg.astype('uint8'))
        out_img.putpalette(cam_palette)
        return out_img
    elif dataset == 'mapillary':
        npimg[npimg == -1] = 65
        color_img = _apply_mapillary_palette(npimg)
        out_img = Image.fromarray(color_img)
        return out_img
    elif dataset == 'aeroscapes':
        out_img = Image.fromarray(npimg.astype('uint8'))
        out_img.putpalette(aeroscapes_palette)
        return out_img
    elif dataset == 'weizmannhorses':
        out_img = Image.fromarray(npimg.astype('uint8'))
        out_img.putpalette(weizhorses_palette)
        return out_img
    elif dataset == 'nyuv2':
        out_img = Image.fromarray(npimg.astype('uint8'))
        out_img.putpalette(nyuv2_palette)
        return out_img
    else:
        raise RuntimeError("Unknown palette for {}".format(dataset))


def _apply_mapillary_palette(image_array):
    with open(mapillary_config()) as config_file:
        config = json.load(config_file)
    labels = config['labels']
    # palette
    color_array = np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8)
    for label_id, label in enumerate(labels):
        color_array[image_array == label_id] = label["color"]
    return color_array


cam_palette = [
    128, 128, 128,  # sky
    128, 0, 0,  # building
    192, 192, 128,  # column_pole
    128, 64, 128,  # road
    0, 0, 192,  # sidewalk
    128, 128, 0,  # tree
    192, 128, 128,  # SignSymbol
    64, 64, 128,  # fence
    64, 0, 128,  # car
    64, 64, 0,  # pedestrian
    0, 128, 192,  # bicyclist
    0, 0, 0  # void
]

aeroscapes_palette = [
    0, 0, 0,  # background
    192, 128, 128,  # person
    0, 128, 0,  # bike
    128, 128, 128,  # car
    128, 0, 0,  # drone
    0, 0, 128,  # boat
    192, 0, 128,  # animal
    192, 0, 0,  # obstacle
    192, 128, 0,  # construction
    0, 64, 0,  # vegetation
    128, 128, 0,  # road
    0, 128, 128  # sky
]

weizhorses_palette = [
    0, 0, 0,
    255, 255, 255
]

nyuv2_palette = [
    # 0, 0, 0,
    255, 20, 23,
    255, 102, 17,
    255, 136, 68,
    255, 238, 85,
    254, 254, 56,
    255, 255, 153,
    170, 204, 34,
    187, 221, 119,
    200, 207, 130,
    146, 167, 126,
    85, 153, 238,
    0, 136, 204,
    34, 102, 136,
    23, 82, 121,
    85, 119, 119,
    221, 187, 51,
    211, 167, 109,
    169, 131, 75,
    118, 118, 118,
    81, 87, 74,
    68, 124, 105,
    116, 196, 147,
    142, 140, 109,
    228, 191, 128,
    233, 215, 142,
    226, 151, 93,
    241, 150, 112,
    225, 101, 82,
    201, 74, 83,
    190, 81, 104,
    163, 73, 116,
    153, 55, 103,
    101, 56, 125,
    78, 36, 114,
    145, 99, 182,
    226, 121, 163,
    224, 89, 139,
    124, 159, 176,
    86, 152, 196,
    154, 191, 136
]
