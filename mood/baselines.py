import optuna
import numpy as np
import datamol as dm

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingRegressor, VotingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import PairwiseKernel, Sum, WhiteKernel
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import ClassifierMixin, clone


"""
These are the three baselines we consider in the MOOD study
For benchmarking, we use a torch MLP rather than the one from scikit-learn.
"""
MOOD_BASELINES = ["MLP", "RF", "GP"]


def get_baseline_cls(name, is_regression):
    """Simple method that allows a model to be identified by its name"""

    target_type = "regression" if is_regression else "classification"
    data = {
        "MLP": {
            "regression": MLPRegressor,
            "classification": MLPClassifier,
        },
        "RF": {
            "regression": RandomForestRegressor,
            "classification": RandomForestClassifier,
        },
        "GP": {
            "regression": GaussianProcessRegressor,
            "classification": GaussianProcessClassifier,
        },
    }

    return data[name][target_type]


def get_baseline_model(
    name: str,
    is_regression: bool,
    params: dict,
    for_uncertainty_estimation: bool = False,
    ensemble_size: int = 10,
    calibrate: bool = True,
):
    """Entrypoint for constructing a baseline model from scikit-learn"""
    model = get_baseline_cls(name, is_regression)(**params)
    if for_uncertainty_estimation:
        model = uncertainty_wrapper(model, ensemble_size, calibrate)
    return model


def uncertainty_wrapper(model, ensemble_size: int = 10, calibrate: bool = True):
    """Wraps the model so that it can be used for uncertainty estimation.
    This includes at most two steps: Turning MLPs in an ensemble and calibrating RF and MLP classifiers
    """
    if isinstance(model, MLPClassifier) or isinstance(model, MLPRegressor):
        models = []
        for idx in range(ensemble_size):
            model = clone(model)
            model.set_params(random_state=model.random_state + idx)
            models.append((f"mlp_{idx}", model))

        if isinstance(model, MLPClassifier):
            model = VotingClassifier(models, voting="soft", n_jobs=-1)
        else:
            model = VotingRegressor(models, n_jobs=-1)

    if calibrate and isinstance(model, RandomForestClassifier) or isinstance(model, MLPClassifier):
        model = CalibratedClassifierCV(model)
    return model


def predict_baseline_uncertainty(model, X):
    """Predicts the uncertainty of the model.

    For GP regressors, we use the included uncertainty estimation.
    For classifiers, the entropy of the prediction is used as uncertainty.
    For regressors, the variance of the prediction is used as uncertainty.
    """
    if isinstance(model, ClassifierMixin):
        uncertainty = model.predict_proba(X)

    elif isinstance(model, GaussianProcessRegressor):
        std = model.predict(X, return_std=True)[1]
        uncertainty = std**2

    else:
        # VotingRegressor or RandomForestRegressor
        preds = dm.utils.parallelized(lambda x: x.predict(X), model.estimators_, n_jobs=model.n_jobs)
        uncertainty = np.var(preds, axis=0)

    return uncertainty


def suggest_mlp_hparams(trial, is_regression):
    """Sample the hyper-parameter search space for MLPs"""
    architectures = [[width] * depth for width in [64, 128, 256] for depth in range(1, 4)]
    arch = trial.suggest_categorical("hidden_layer_sizes", architectures)
    lr = trial.suggest_float("learning_rate_init", 1e-7, 1e0, log=True)
    alpha = trial.suggest_float("alpha", 1e-10, 1e0, log=True)
    max_iter = trial.suggest_int("max_iter", 1, 300)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    return {
        "max_iter": max_iter,
        "alpha": alpha,
        "learning_rate_init": lr,
        "hidden_layer_sizes": arch,
        "batch_size": batch_size,
    }


def suggest_rf_hparams(trial, is_regression):
    """Sample the hyper-parameter search space for RFs"""
    n_estimators = trial.suggest_int("n_estimators", 100, 1000)
    max_depth = trial.suggest_categorical("max_depth", [None] + list(range(1, 11)))

    params = {
        "max_depth": max_depth,
        "n_estimators": n_estimators,
    }
    return params


def construct_kernel(is_regression, params):
    """Constructs a scikit-learn kernel based on provided hyper-parameters"""

    metric = params.pop("kernel_metric")
    gamma = params.pop("kernel_gamma")
    coef0 = params.pop("kernel_coef0")

    if is_regression:
        kernel = PairwiseKernel(metric=metric, gamma=gamma, pairwise_kernels_kwargs={"coef0": coef0})
    else:
        noise_level = params.pop("kernel_noise_level")
        kernel = Sum(
            PairwiseKernel(
                metric=metric,
                gamma=gamma,
                pairwise_kernels_kwargs={"coef0": coef0},
            ),
            WhiteKernel(noise_level=noise_level),
        )
    return kernel, params


def suggest_gp_hparams(trial, is_regression):
    """Sample the hyper-parameter search space for GPs"""

    kernel_types = ["linear", "poly", "polynomial", "rbf", "laplacian", "sigmoid", "cosine"]
    metric = trial.suggest_categorical("kernel_metric", kernel_types)
    gamma = trial.suggest_float("kernel_gamma", 1e-5, 1e0, log=True)
    coef0 = trial.suggest_float("kernel_coef0", 1e-5, 1.0, log=True)
    params = {"kernel_gamma": gamma, "kernel_coef0": coef0, "kernel_metric": metric}

    n_restarts_optimizer = trial.suggest_int("n_restarts_optimizer", 0, 10)
    params["n_restarts_optimizer"] = n_restarts_optimizer

    if is_regression:
        params["alpha"] = trial.suggest_float("alpha", 1e-10, 1e0, log=True)
    else:
        max_iter_predict = trial.suggest_int("max_iter_predict", 10, 250)
        noise_level = trial.suggest_float("kernel_noise_level", 1e-5, 1, log=True)
        params["kernel_noise_level"] = noise_level
        params["max_iter_predict"] = max_iter_predict

    return params


def suggest_baseline_hparams(name: str, is_regression: bool, trial: optuna.Trial):
    """Endpoint for sampling the hyper-parameter search space of the baselines"""
    fs = {
        "MLP": suggest_mlp_hparams,
        "RF": suggest_rf_hparams,
        "GP": suggest_gp_hparams,
    }
    return fs[name](trial, is_regression)
