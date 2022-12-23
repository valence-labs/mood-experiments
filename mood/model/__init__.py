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


MOOD_ALGORITHMS = {
    "VREx": VREx,
    "CORAL": CORAL,
    "DANN": DANN,
    "IB-ERM": InformationBottleneckERM,
    "Mixup": Mixup,
    "MLP": ERM,
    "MTL": MTL
}


def is_domain_adaptation(model: Union[BaseEstimator, BaseModel]):
    if not (isinstance(model, BaseEstimator) or isinstance(model, BaseModel)):
        raise TypeError(f"Can only test models from sklearn, good-learn or mood, not {type(model)}")
    return isinstance(model, Mixup) or isinstance(model, DANN) or isinstance(model, CORAL)


def is_domain_generalization(model: Union[BaseEstimator, BaseModel]):
    if not (isinstance(model, BaseEstimator) or isinstance(model, BaseModel)):
        raise TypeError(f"Can only test models from sklearn, good-learn or mood, not {type(model)}")
    return isinstance(model, MTL) or isinstance(model, InformationBottleneckERM) or isinstance(model, VREx)
