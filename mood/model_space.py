import datamol as dm
import numpy as np

from typing import Union, Optional

from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import PairwiseKernel

from sklearn.preprocessing import OneHotEncoder


_SKLEARN_MLP_TYPE = Union[MLPRegressor, MLPClassifier]
_SKLEARN_RF_TYPE = Union[RandomForestRegressor, RandomForestClassifier]
_SKLEARN_GP_TYPE = Union[GaussianProcessRegressor, GaussianProcessClassifier]


class ModelSpaceTransformer:
    
    SUPPORTED_TYPES = Union[
        _SKLEARN_MLP_TYPE, 
        _SKLEARN_RF_TYPE, 
        _SKLEARN_GP_TYPE,
    ]
        
    def __init__(self, model, n_jobs: Optional[int] = None):
        if not isinstance(model, self.SUPPORTED_TYPES):
            raise TypeError(f"{type(model)} is not supported")
        self._model = model
        self._n_jobs = n_jobs
    
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
        
    @staticmethod
    def get_rf_embedding(model, X):
        """
        For a random forest, the model space embedding is given by
        a subset of 100 features that have the highest importance 
        """            
        importances = model.feature_importances_
        mask = np.argsort(importances)[-100:]
        return X[:, mask]
    
    @staticmethod
    def get_gp_embedding(model, X):
        """
        In a Gaussian Process, the model space embedding is given
        by a subset of 100 features that have the highest importance. 
        This importance is computed based on alpha and the train set. 
        For now, this only supports a linear kernel.
        """
        if not (isinstance(model.kernel_, PairwiseKernel) and model.kernel_.metric == "linear"):
            msg = f"We only support PairwiseKernel(metric='linear'), not {model.kernel_}"
            raise NotImplementedError(msg)
        importances = (model.alpha_[:, None] * model.X_train_).sum(axis=0)
        importances = np.abs(importances)
        mask = np.argsort(importances)[-100:]
        return X[:, mask]
    
    @staticmethod
    def get_mlp_embedding(model, X):
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
        return activations[-2]
