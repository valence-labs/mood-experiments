from typing import Union, Callable

import six
import torch
from sklearn.base import BaseEstimator

from good.base import BaseModel, BasePTLModel
from good.kernels.base import OODKernelMethod
from mood.model.coral import CORAL
from mood.model.ib_erm import InformationBottleneckERM
from mood.model.mixup import Mixup
from mood.model.dann import DANN
from mood.model.erm import ERM
from mood.model.mtl import MTL
from mood.model.vrex import VREx


def is_ood(model: Union[BaseEstimator, BaseModel]):
    if not (isinstance(model, BaseEstimator) or isinstance(model, BaseModel)):
        raise TypeError(f"Can only test models from sklearn, good-learn or mood, not {type(model)}")
    is_ood_kernel = isinstance(model, OODKernelMethod)
    is_ood_torch = isinstance(model, BasePTLModel) and not isinstance(model, ERM)
    return is_ood_kernel or is_ood_torch


def is_domain_adaptation(model: Union[BaseEstimator, BaseModel]):
    if not (isinstance(model, BaseEstimator) or isinstance(model, BaseModel)):
        raise TypeError(f"Can only test models from sklearn, good-learn or mood, not {type(model)}")
    return isinstance(model, Mixup) or isinstance(model, DANN) or isinstance(model, CORAL)


def is_domain_generalization(model: Union[BaseEstimator, BaseModel]):
    if not (isinstance(model, BaseEstimator) or isinstance(model, BaseModel)):
        raise TypeError(f"Can only test models from sklearn, good-learn or mood, not {type(model)}")
    return isinstance(model, MTL) or isinstance(model, InformationBottleneckERM) or isinstance(model, VREx)


def linear_interpolation(step, duration, max_value, start: int = 0, min_value: float = 0):
    if max_value < min_value:
        raise ValueError("max_value cannot be smaller than min value")
    if step < start:
        return min_value
    if step >= start + duration:
        return max_value
    step_size = (max_value - min_value) / duration
    step = step - start
    return step_size * step
