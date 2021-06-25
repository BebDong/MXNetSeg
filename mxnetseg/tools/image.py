# coding=utf-8

import os
import numpy as np
from PIL import Image
from mxnet import nd
from mxnet.gluon.data.vision import transforms
from typing import List, Union, Tuple, NamedTuple
from .path import demo_dir

RGB_MEAN = (123.68, 116.779, 103.939)
RGB_MEAN_NORM = (0.485, 0.456, 0.406)
RGB_STD = (58.393, 57.12, 57.375)
RGB_STD_NORM = (0.229, 0.224, 0.225)


class _ImageShape(NamedTuple):
    w: int
    h: int
    c: int = 3


def image_transform():
    """to tensor and normalize"""
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(RGB_MEAN_NORM, RGB_STD_NORM)])
    return trans


def apply_transform(img: np.ndarray, my_trans=None) -> nd.NDArray:
    """apply default or self-defined transform"""
    trans = my_trans if my_trans else image_transform()
    img = trans(nd.array(img))
    img = nd.expand_dims(img, axis=0)
    return img


def _available_images(path: str) -> List[str]:
    """
    List available image names under path directory.
    """
    images = os.listdir(path)
    return [im.replace('.png', '') for im in images if '.png' in im] + \
           [im.replace('.jpg', '') for im in images if '.jpg' in im]


def get_demo_image_path(name: str, tag: str = 'affinity') -> str:
    """
    get image path under /demo/tag/ directory by image name
    """
    if name not in _available_images(os.path.join(demo_dir(), tag)):
        raise FileNotFoundError(f"Image does not exist: {name}")

    image_path_png = os.path.join(demo_dir(), tag, f'{name}.png')
    image_path_jpg = os.path.join(demo_dir(), tag, f'{name}.jpg')
    png_exist = os.path.exists(image_path_png)
    jpg_exist = os.path.exists(image_path_jpg)
    if not any([png_exist, jpg_exist]):
        raise FileNotFoundError(f"Only support .png/.jpg format: {name}")
    image_path = image_path_png if png_exist else image_path_jpg
    return image_path


def load_image(img_path: str, shape: Tuple[int, int, int] = None,
               as_image=False) -> Union[np.ndarray, Image.Image]:
    """
    load image from a specific path.

    :param img_path: image path
    :param shape: (width, height, channels)
    :param as_image: as PIL.Image.Image
    :return: numpy array of shape [height, width, 3] if as_image=False,
        otherwise PIL Image object
    """
    img = Image.open(img_path, mode='r').convert('RGB')
    if shape:
        shape = _ImageShape(*shape)
        img = img.resize((shape.w, shape.h), Image.ANTIALIAS)
    if as_image:
        return img
    return np.array(img)


def load_image_mask(img_name, tag: str = 'affinity', shape: Tuple[int, int, int] = None,
                    as_image=False):
    """
    get image and corresponding segmentation mask.

    :param img_name: image name
    :param tag: sub-directory of the demo folder
    :param shape: image shape (width, height, channels)
    :param as_image: if True, function returns PIL Image object, else
        numpy array.
    :return: numpy array of shape [height, width, 3] if as_image=False,
        otherwise PIL Image object
    """
    img = Image.open(get_demo_image_path(img_name, tag), mode='r').convert('RGB')
    mask = Image.open(get_demo_image_path(name=f'{img_name}.mask', tag=tag))
    if shape:
        shape = _ImageShape(*shape)
        img = img.resize((shape.w, shape.h), Image.ANTIALIAS)
        mask = mask.resize((shape.w, shape.h), Image.NEAREST)
    if as_image:
        return img, mask
    return np.array(img), np.array(mask)


def mask_transform(mask: np.ndarray, ignore_class: int = 0, one_hot: bool = False, n_class: int = None) -> nd.NDArray:
    """
    transform mask to NxCxHxW shape

    :param mask: numpy array mask with shape (height, width)
    :param ignore_class: set ignore_label to -1
    :param one_hot: transfer the mask to one-hot. When true, need to give corresponding n_class.
    :param n_class: number of classes
    :return:
    """
    assert len(mask.shape) == 2, "Must be a single channel mask instead of a RGB one"
    if ignore_class == 0:
        mask = mask - 1
    else:
        mask[mask == ignore_class] = -1

    mask = nd.array(mask)
    if one_hot:
        mask = nd.one_hot(mask, depth=n_class)
        mask = mask.astype('float32')
        mask = nd.transpose(mask, axes=(2, 0, 1))
    return mask.expand_dims(0)


def denormalize(img: nd.NDArray) -> nd.NDArray:
    """
    reverse the normalization on an image
    normalization: (img - mean) / std
    de-normalization: img * std + mean

    :param img: with shape NCHW
    :return: de-normalized image with pixel values in [0, 1]
    """
    mean = nd.array(RGB_MEAN_NORM)
    std = nd.array(RGB_STD_NORM)
    denormalized = img * std.reshape(shape=(3, 1, 1)) + mean.reshape(shape=(3, 1, 1))
    return denormalized


def mask_color_to_gray(mask_gray_dir, mask_color_dir):
    """convert color mask to gray"""
    if not os.path.exists(mask_gray_dir):
        os.makedirs(mask_gray_dir)

    for ann in os.listdir(mask_color_dir):
        ann_im = Image.open(os.path.join(mask_color_dir, ann))
        ann_im = Image.fromarray(np.array(ann_im))
        ann_im.save(os.path.join(mask_gray_dir, ann))
