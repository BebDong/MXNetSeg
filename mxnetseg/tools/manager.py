# coding=utf-8
# Adapted from:
# https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.1/paddleseg/cvlibs/manager.py

import inspect
from collections.abc import Sequence

__all__ = ['ComponentManager', 'BACKBONES', 'MODELS', 'DATASETS', 'LOSSES']


class ComponentManager:
    def __init__(self, name=None):
        self._name = name
        self._components_dict = dict()

    def __len__(self):
        return len(self._components_dict)

    def __repr__(self):
        name_str = self._name if self._name else self.__class__.__name__
        return "{}:{}".format(name_str, list(self._components_dict.keys()))

    def __getitem__(self, item):
        if item not in self._components_dict.keys():
            raise KeyError("{} does not exist in available {}".format(item, self))
        return self._components_dict[item]

    @property
    def components_dict(self):
        return self._components_dict

    @property
    def name(self):
        return self._name

    def _add_single_component(self, component):
        if not (inspect.isclass(component) or inspect.isfunction(component)):
            raise TypeError("Expect class/function type, but received {}".format(type(component)))
        component_name = component.__name__
        if component_name in self._components_dict.keys():
            raise KeyError("{} exists already!".format(component_name))
        else:
            self._components_dict[component_name] = component

    def add_component(self, components):
        if isinstance(components, Sequence):
            for component in components:
                self._add_single_component(component)
        else:
            self._add_single_component(components)
        return components


BACKBONES = ComponentManager('backbones')
MODELS = ComponentManager('models')
DATASETS = ComponentManager('datasets')
LOSSES = ComponentManager('losses')
