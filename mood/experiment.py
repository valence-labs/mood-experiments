import numpy as np
import optuna
import torch
from loguru import logger
from sklearn.base import BaseEstimator

from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from mood.baselines import suggest_baseline_hparams, predict_baseline_uncertainty, MOOD_BASELINES
from mood.model import MOOD_ALGORITHMS
from mood.model.base import Ensemble
from mood.train import train_baseline_model, train
from mood.criteria import get_mood_criteria
from mood.dataset import load_data_from_tdc, SimpleMolecularDataset, MOOD_REGR_DATASETS
from mood.distance import get_distance_metric
from mood.metrics import Metric
from mood.representations import featurize
from mood.preprocessing import DEFAULT_PREPROCESSING
from mood.splitter import get_mood_splitters, MOODSplitter
from mood.utils import load_distances_for_downstream_application
from mood.rct import get_experimental_configurations


def get_predictions(model, dataset):
    if isinstance(model, BaseEstimator):
        y_pred = model.predict(dataset.X).reshape(-1, 1)
        uncertainty = predict_baseline_uncertainty(model, dataset.X)
    elif isinstance(model, Ensemble):
        dataloader = DataLoader(dataset)
        y_pred = model.predict(dataloader)
        uncertainty = model.predict_uncertainty(dataloader)
    else:
        raise NotImplementedError
    return y_pred, uncertainty


def basic_tuning_loop(
    X_train,
    X_test,
    y_train,
    y_test,
    name: str,
    is_regression: bool,
    metric: Metric,
    global_seed: int,
    for_uncertainty_estimation: bool = False,
    ensemble_size: int = 10,
    n_trials: int = 100,
    n_startup_trials: int = 20,
):

    # NOTE: This could be merged with the more elaborate tuning loop we wrote later
    #   However, for the sake of reproducibility, I wanted to keep this code intact.
    #   This way, the exact code used to generate results is still easily accessible
    #   in the code base

    def run_trial(trial):
        random_state = global_seed + trial.number
        params = suggest_baseline_hparams(name, is_regression, trial)
        model = train_baseline_model(
            X_train,
            y_train,
            name,
            is_regression,
            params,
            random_state,
            for_uncertainty_estimation,
            ensemble_size,
        )
        y_pred = model.predict(X_test)
        score = metric(y_test, y_pred)
        return score

    direction = "maximize" if metric.mode == "max" else "minimize"
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


def rct_tuning_loop(
    train_val_dataset: SimpleMolecularDataset,
    test_dataset: SimpleMolecularDataset,
    algorithm: str,
    train_val_split: str,
    criterion_name: str,
    performance_metric: Metric,
    calibration_metric: Metric,
    is_regression: bool,
    global_seed: int,
    num_repeated_splits: int = 5,
    num_trials: int = 50,
    num_startup_trials: int = 10,
):

    def run_trial(trial):

        random_state = global_seed + trial.number

        splitters = get_mood_splitters(
            train_val_dataset.smiles,
            num_repeated_splits,
            random_state,
            n_jobs=-1
        )
        train_val_splitter = splitters[train_val_split]

        for train_ind, val_ind in train_val_splitter.split(train_val_dataset.X):

            train_dataset = train_val_dataset.filter_by_indices(train_ind)
            val_dataset = train_val_dataset.filter_by_indices(val_ind)

            # Z-standardization of the targets
            if is_regression:
                scaler = StandardScaler()
                train_val_dataset.y = scaler.fit_transform(train_val_dataset.y)
                test_dataset.y = scaler.transform(test_dataset.y)

            random_state = global_seed + trial.number

            if algorithm in MOOD_ALGORITHMS:
                params = MOOD_ALGORITHMS[algorithm].suggest_params(trial)
            elif algorithm in MOOD_BASELINES:
                params = suggest_baseline_hparams(algorithm, is_regression, trial)
            else:
                raise NotImplementedError

            model = train(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                algorithm=algorithm,
                is_regression=is_regression,
                params=params,
                seed=random_state,
            )
            y_pred, uncertainty = get_predictions(model, val_dataset)
            criterion.update(y_pred, uncertainty, train_dataset, val_dataset)

        return criterion.critique()

    criterion = get_mood_criteria(performance_metric, calibration_metric)[criterion_name]

    direction = "maximize" if criterion.mode == "max" else "minimize"
    sampler = optuna.samplers.TPESampler(seed=global_seed, n_startup_trials=num_startup_trials)
    study = optuna.create_study(direction=direction, sampler=sampler)

    # ValueError: array must not contain infs or NaNs
    # LinAlgError: N-th leading minor of the array is not positive definite
    # LinAlgError: The kernel is not returning a positive definite matrix
    if algorithm == "GP":
        catch = (np.linalg.LinAlgError, ValueError)
    else:
        catch = ()

    study.optimize(run_trial, n_trials=num_trials, catch=catch)

    random_state = study.best_trial.number
    splitters = get_mood_splitters(
        train_val_dataset.smiles,
        num_repeated_splits,
        random_state,
        n_jobs=-1
    )
    train_val_splitter = splitters[train_val_split]
    train_ind, val_ind = next(train_val_splitter.split(train_val_dataset.X))
    train_dataset = train_val_dataset.filter_by_indices(train_ind)
    val_dataset = train_val_dataset.filter_by_indices(val_ind)

    # Z-standardization of the targets
    if is_regression:
        scaler = StandardScaler()
        train_val_dataset.y = scaler.fit_transform(train_val_dataset.y)
        test_dataset.y = scaler.transform(test_dataset.y)

    model = train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        algorithm=algorithm,
        is_regression=is_regression,
        params=study.best_params,
        seed=random_state,
    )

    y_pred, uncertainty = get_predictions(model, test_dataset)
    cal_score = calibration_metric(test_dataset.y, y_pred, uncertainty)
    prf_score = performance_metric(test_dataset.y, y_pred, uncertainty)
    return study, cal_score, prf_score


def tune_cmd(dataset, algorithm, representation, train_val_split, criterion, seed: int = 0, use_cache: bool = False):

    smiles, y = load_data_from_tdc(dataset, disable_logs=True)
    X, mask = featurize(smiles, representation, DEFAULT_PREPROCESSING[representation], disable_logs=True)
    X = X.astype(np.float32)
    smiles = smiles[mask]
    y = y[mask]

    is_regression = dataset in MOOD_REGR_DATASETS
    if is_regression:
        y = y.reshape(-1, 1)

    distances_vs = load_distances_for_downstream_application(
        "virtual_screening", representation, dataset, update_cache=not use_cache
    )
    distances_op = load_distances_for_downstream_application(
        "optimization", representation, dataset, update_cache=not use_cache
    )

    distance_metric = get_distance_metric(X)
    splitters = get_mood_splitters(smiles, 5, seed, n_jobs=-1)
    train_test_splitter = MOODSplitter(splitters, np.concatenate((distances_vs, distances_op)), distance_metric, k=5)
    train_test_splitter.fit(X)

    # Use the prescribed split
    trainval, test = next(train_test_splitter.split(X, y))
    train_val_dataset = SimpleMolecularDataset(smiles[trainval], X[trainval], y[trainval])
    test_dataset = SimpleMolecularDataset(smiles[test], X[test], y[test])

    study, cal_score, prf_score = rct_tuning_loop(
        train_val_dataset=train_val_dataset,
        test_dataset=test_dataset,
        algorithm=algorithm,
        train_val_split=train_val_split,
        criterion_name=criterion,
        performance_metric=Metric.get_default_performance_metric(dataset),
        calibration_metric=Metric.get_default_calibration_metric(dataset),
        is_regression=is_regression,
        global_seed=seed,
    )

    print(study.best_value, cal_score, prf_score)


def rct_cmd(dataset: str, index: int):
    configs = get_experimental_configurations(dataset)
    logger.info(f"Sampled configuration #{index} / {len(configs)} for {dataset}: {configs[index]}")
    algorithm, representation, train_val_split, criterion, seed = configs[index]
    tune_cmd(dataset, algorithm, representation, train_val_split, criterion, seed)
