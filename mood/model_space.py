import numpy as np

from typing import Union

from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import PairwiseKernel, Sum


_SKLEARN_MLP_TYPE = Union[MLPRegressor, MLPClassifier]
_SKLEARN_RF_TYPE = Union[RandomForestRegressor, RandomForestClassifier]
_SKLEARN_GP_TYPE = Union[GaussianProcessRegressor, GaussianProcessClassifier]


def is_linear_kernel(kernel):
    return isinstance(kernel, PairwiseKernel) and kernel.metric == "linear"


class ModelSpaceTransformer:

    SUPPORTED_TYPES = Union[
        _SKLEARN_MLP_TYPE,
        _SKLEARN_RF_TYPE,
        _SKLEARN_GP_TYPE,
    ]

    def __init__(self, model, embedding_size: int):
        if not isinstance(model, self.SUPPORTED_TYPES):
            raise TypeError(f"{type(model)} is not supported")
        self._model = model
        self._embedding_size = embedding_size

    def __call__(self, X):
        """Transforms a list of datapoints"""
        return self.transform(X)

    def transform(self, X):
        """Transforms a single datapoint"""
        if isinstance(self._model, _SKLEARN_RF_TYPE):
            return self.get_rf_embedding(self._model, X)
        elif isinstance(self._model, _SKLEARN_GP_TYPE):
            return self.get_gp_embedding(self._model, X)
        elif isinstance(self._model, _SKLEARN_MLP_TYPE):
            return self.get_mlp_embedding(self._model, X)
        # This should never be reached given the
        # type check in the constructor
        raise NotImplementedError

    def get_rf_embedding(self, model, X):
        """
        For a random forest, the model space embedding is given by
        a subset of 100 features that have the highest importance
        """
        importances = model.feature_importances_
        mask = np.argsort(importances)[-self._embedding_size :]
        return X[:, mask]

    def get_gp_embedding(self, model, X):
        """
        In a Gaussian Process, the model space embedding is given
        by a subset of 100 features that have the highest importance.
        This importance is computed based on alpha and the train set.
        For now, this only supports a linear kernel.
        """
        # Check the target type
        is_regression = isinstance(model, GaussianProcessRegressor)
        if not is_regression and model.n_classes_ != 2:
            msg = f"We only support regression and binary classification"
            raise ValueError(msg)

        # Check the kernel type
        is_linear = (
            is_linear_kernel(model.kernel_)
            or isinstance(model.kernel_, Sum)
            and (is_linear_kernel(model.kernel_.k1) or is_linear_kernel(model.kernel_.k2))
        )
        if not is_linear:
            msg = f"We only support the linear kernel, not {model.kernel_}"
            raise NotImplementedError(msg)

        if is_regression:
            alpha = model.alpha_
            X_train = model.X_train_
        else:
            est = model.base_estimator_
            alpha = est.y_train_ - est.pi_
            X_train = est.X_train_

        importances = (alpha[:, None] * X_train).sum(axis=0)
        importances = np.abs(importances)
        mask = np.argsort(importances)[-self._embedding_size :]
        return X[:, mask]

    def get_mlp_embedding(self, model, X):
        """
        For an multi-layer perceptron, the model space embedding is given by
        the activations of the second-to-last layer
        """
        hidden_layer_sizes = model.hidden_layer_sizes

        # Get the MLP architecture
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)
        layer_units = [X.shape[1]] + hidden_layer_sizes + [model.n_outputs_]

        # Create empty arrays to save all activations in
        activations = [X]
        for i in range(model.n_layers_ - 1):
            activations.append(np.empty((X.shape[0], layer_units[i + 1])))

        # Actually populate the empty arrays
        model._forward_pass(activations)

        # Return the activations of the second-to-last layer
        hidden_rep = activations[-2]

        importances = model.coefs_[-1][:, 0]
        mask = np.argsort(importances)[-self._embedding_size :]
        return hidden_rep[:, mask]
