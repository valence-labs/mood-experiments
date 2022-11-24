import optuna
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import PairwiseKernel, Sum, WhiteKernel

from mood.metrics import compute_metric, get_metric_direction


SUPPORTED_BASELINES = ["MLP", "RF", "GP"]


def get_baseline_cls(name, is_regression):

    target_type = "regression" if is_regression else "classification"
    data = {
        "MLP": {
            "regression":  MLPRegressor,
            "classification": MLPClassifier,
        },
        "RF": {
            "regression":  RandomForestRegressor,
            "classification": RandomForestClassifier,
        },
        "GP": {
            "regression":  GaussianProcessRegressor,
            "classification": GaussianProcessClassifier,
        }
    }
    
    return data[name][target_type]

                         
def get_baseline_model(name, is_regression, params, seed):
    return get_baseline_cls(name, is_regression)(**params)

    
def train_model(X, y, name: str, is_regression: bool, params: dict, seed: int):
    params["random_state"] = seed
    
    if name == "RF" and not is_regression:
        params["class_weight"] = "balanced"
    if name == "GP":
        params["kernel"], params = construct_kernel(is_regression, params)
        
    model = get_baseline_model(name, is_regression, params, seed)
    model.fit(X, y)
    return model


def suggest_mlp_hparams(trial, is_regression):
       
    architectures = [[width] * depth for width in [64, 128, 256] for depth in range(1, 4)]
    arch = trial.suggest_categorical('hidden_layer_sizes', architectures)
    lr = trial.suggest_float('learning_rate_init', 1e-7, 1e0, log=True)
    alpha = trial.suggest_float('alpha', 1e-10, 1e0, log=True)
    max_iter = trial.suggest_int('max_iter', 1, 300)
    
    return {
        "max_iter": max_iter, 
        "alpha": alpha, 
        "learning_rate_init": lr, 
        "hidden_layer_sizes": arch, 
    }


def suggest_rf_hparams(trial, is_regression):
    
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    max_depth = trial.suggest_categorical('max_depth', [None] + list(range(1, 11)))

    params = {
        "max_depth": max_depth, 
        "n_estimators": n_estimators, 
    }       
    return params


def construct_kernel(is_regression, params):
    
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
            WhiteKernel(noise_level=noise_level)
        )
    return kernel, params


def suggest_gp_hparams(trial, is_regression):
    
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
    fs = {
        "MLP": suggest_mlp_hparams,
        "RF": suggest_rf_hparams,
        "GP": suggest_gp_hparams,
    }
    return fs[name](trial, is_regression)


def tune_model(
    X_train, 
    X_test, 
    y_train, 
    y_test, 
    name: str, 
    is_regression: bool, 
    metric: str, 
    global_seed: int,
    n_trials: int = 100,
    n_startup_trials: int = 20,
):

    def run_trial(trial):
        random_state = global_seed + trial.number
        params = suggest_baseline_hparams(name, is_regression, trial)
        model = train_model(X_train, y_train, name, is_regression, params, random_state)
        y_pred = model.predict(X_test)
        score = compute_metric(y_pred, y_test, metric)
        return score
    
    direction = get_metric_direction(metric)
    sampler = optuna.samplers.TPESampler(seed=global_seed, n_startup_trials=n_startup_trials)
    
    study = optuna.create_study(direction=direction, sampler=sampler)
    
    # ValueError: array must not contain infs or NaNs
    # LinAlgError: N-th leading minor of the array is not positive definite
    # LinAlgError: The kernel is not returning a positive definite matrix
    if name == "GP":
        catch = (np.linalg.LinAlgError, ValueError)
    else: 
        catch = ()
        
    study.optimize(run_trial, n_trials=n_trials, catch=catch)
    return study
