from typing import Union

from sklearn.base import BaseEstimator

from mood.model.base import BaseModel
from mood.model.vrex import VREx
from mood.model.coral import CORAL
from mood.model.dann import DANN
from mood.model.ib_erm import InformationBottleneckERM
from mood.model.mixup import Mixup
from mood.model.erm import ERM
from mood.model.mtl import MTL


MOOD_DA_DG_ALGORITHMS = {
    "VREx": VREx,
    "CORAL": CORAL,
    "DANN": DANN,
    "IB-ERM": InformationBottleneckERM,
    "Mixup": Mixup,
    "MLP": ERM,
    "MTL": MTL,
}

MOOD_ALGORITHMS = [
    "RF",
    "GP",
    "MLP",
    "MTL",
    "VREx",
    "IB-ERM",
    "CORAL",
    "DANN",
    "Mixup"
]


def _get_type(model: Union[BaseEstimator, BaseModel, str]):
    if isinstance(model, str):
        model_type = MOOD_DA_DG_ALGORITHMS.get(model)
    else:
        model_type = type(model)
    if not (model_type is None or issubclass(model_type, BaseEstimator) or issubclass(model_type, BaseModel)):
        raise TypeError(f"Can only test models from sklearn, good-learn or mood, not {model_type}")
    return model_type


def is_domain_adaptation(model: Union[BaseEstimator, BaseModel, str]):
    model_type = _get_type(model)
    return model_type in [Mixup, DANN, CORAL]


def is_domain_generalization(model: Union[BaseEstimator, BaseModel, str]):
    model_type = _get_type(model)
    return model_type in [MTL, InformationBottleneckERM, VREx]


def needs_domain_representation(model: Union[BaseEstimator, BaseModel, str]):
    model_type = _get_type(model)
    return model_type == MTL
