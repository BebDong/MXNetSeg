# coding=utf-8

import os
import shutil
import platform


def data_dir():
    """data directory in the filesystem for model storage,
    for example when downloading models"""
    return os.getenv('MXNET_HOME', _data_dir_default())


def _data_dir_default():
    """default data directory depending on the platform and environment variables"""
    system = platform.system()
    if system == 'Windows':
        return os.path.join(os.environ.get('APPDATA'), 'mxnet')
    else:
        return os.path.join(os.path.expanduser("~"), '.mxnet')


def makedir_p(*paths):
    """mkdir -p"""
    path = os.path.join(*paths)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def root_dir():
    """root dir"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def dataset_dir():
    """dataset dir"""
    return os.path.join(root_dir(), 'dataset')


def demo_dir():
    """demo dir"""
    return os.path.join(root_dir(), 'demo')


def wandb_dir():
    """wandb dir"""
    return os.path.join(root_dir(), 'wandb')


def _weights_dir():
    """weights dir for saving models' weights"""
    return os.path.join(root_dir(), 'weights')


def weight_dir(model_name: str):
    """dir for saving weights of a specific model given by name"""
    return makedir_p(_weights_dir(), model_name.lower())


def mapillary_config():
    """config file for Mapillary segmentation dataset"""
    return os.path.join(dataset_dir(), 'Mapillary', 'config.json')


def imagenet_rec():
    """ImageNet dataset with ImageRecord format"""
    return os.path.join(dataset_dir(), 'ImageNet', 'rec')


def validate_checkpoint(model_name: str, checkpoint: str):
    checkpoint = os.path.join(weight_dir(model_name.lower()), checkpoint)
    if not os.path.isfile(checkpoint):
        raise RuntimeError(f"No model params found at {checkpoint}")
    return checkpoint


def save_checkpoint(model, model_name, backbone, data_name, time_stamp, is_best=False):
    if backbone:
        filename = "%s_%s_%s_%s.params" % (model_name, backbone, data_name, time_stamp)
    else:
        filename = "%s_%s_%s.params" % (model_name, data_name, time_stamp)
    filepath = os.path.join(weight_dir(model_name), filename)
    model.save_parameters(filepath)
    if is_best:
        shutil.copyfile(filepath, filepath.replace('.params', '_best.params'))
