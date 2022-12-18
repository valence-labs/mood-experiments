import numpy as np

from typing import Union, Optional

from scipy.spatial.distance import cdist
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier


_SKLEARN_MLP_TYPE = Union[MLPRegressor, MLPClassifier]
_SKLEARN_RF_TYPE = Union[RandomForestRegressor, RandomForestClassifier]
_SKLEARN_GP_TYPE = Union[GaussianProcessRegressor, GaussianProcessClassifier]


class EmpiricalKernelMapTransformer:
    def __init__(self, n_samples: int, metric: str, random_state: Optional[int] = None):
        self._n_samples = n_samples
        self._random_state = random_state
        self._samples = None
        self._metric = metric

    def __call__(self, X):
        """Transforms a list of datapoints"""
        return self.transform(X)

    def transform(self, X):
        """Transforms a single datapoint"""
        if self._samples is None:
            rng = np.random.default_rng(self._random_state)
            self._samples = X[rng.choice(np.arange(len(X)), self._n_samples)]
        X = cdist(X, self._samples, metric=self._metric)
        return X
