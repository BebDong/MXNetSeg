# coding=utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from pylab import axis, title
from mxnet import nd
from typing import Union


def get_row_col(num_pic: int):
    """
    get figure row and column number
    """
    sqr = num_pic ** 0.5
    row = round(sqr)
    col = row + 1 if sqr - row > 0 else row
    return row, col


def format_to_plot(tensor: nd.NDArray) -> np.ndarray:
    """
    format the input tensor from NCHW/CHW to HWC/HW for plotting
    """
    if len(tensor.shape) == 4:
        tensor = nd.squeeze(tensor, axis=0)
    if tensor.shape[0] == 1:
        tensor = nd.squeeze(tensor, axis=0)
    else:
        tensor = nd.transpose(tensor, axes=(1, 2, 0))
    return tensor.asnumpy()


def plot_features(features: nd.NDArray, scope: int = 9):
    """
    visualize feature maps per channel.

    :param features: feature map with shape 1xCxHxW or CxHxW
    :param scope: the index of feature maps to visualize is [0, scope)
    """
    if len(features.shape) == 4:
        scope = scope if scope < features.shape[1] else features.shape[1]
        feature_maps = nd.squeeze(features, axis=0).asnumpy()
    else:
        scope = scope if scope < features.shape[0] else features.shape[0]
        feature_maps = features.asnumpy()
    feature_map_combination = []
    # separate visualization
    row, col = get_row_col(scope)
    plt.figure()
    for i in range(0, scope):
        feature_map = feature_maps[i, :, :]
        feature_map_combination.append(feature_map)
        plt.subplot(row, col, i + 1)
        plt.imshow(feature_map)
        axis('off')
        # title(f"feature map {i}")
    plt.show()
    # overlaps
    feature_map_sum = sum(ele for ele in feature_map_combination)
    plt.imshow(feature_map_sum)
    axis('off')
    plt.show()


def overlap_heat_map(img_path: str, heat_map: Union[np.ndarray, nd.NDArray],
                     alpha: float = 0.4, save_name=None):
    """
    overlap the original image and heat-map.
    """
    import cv2
    if isinstance(heat_map, nd.NDArray):
        heat_map = heat_map.asnumpy()
    assert len(heat_map.shape) == 2
    img = cv2.imread(img_path)
    heat_map = cv2.cvtColor(heat_map, cv2.COLOR_GRAY2BGR)
    heat_map = cv2.resize(heat_map, (img.shape[1], img.shape[0]))
    heat_map = np.uint8(255 * heat_map)
    heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
    overlap = heat_map * alpha + img
    if save_name:
        cv2.imwrite(os.path.join(os.path.dirname(img_path), save_name + '.png'), overlap)
    return overlap


def plot_heatmap(heatmap: Union[nd.NDArray, np.ndarray], fig_width=16,
                 save_pth: str = None, **kwargs):
    """
    plot heat-map which keeps height-width ratio

    :param heatmap: array of shape HW
    :param fig_width: width of figure
    :param save_pth: save plot when not none
    :return:
    """
    if isinstance(heatmap, nd.NDArray):
        heatmap = heatmap.asnumpy()
    assert len(heatmap.shape) == 2, "expect shape of HW"

    h, w = heatmap.shape
    ratio = h / w
    plt.figure(figsize=(fig_width, int(fig_width * ratio)))

    seaborn.set()
    seaborn.heatmap(heatmap, **kwargs)

    plt.axis('off')

    if save_pth:
        dir_path, _ = os.path.split(save_pth)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        plt.savefig(save_pth, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()
