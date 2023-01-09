import fsspec
import optuna
import numpy as np
import datamol as dm

from datetime import datetime
from typing import Optional

import yaml
from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from mood.baselines import suggest_baseline_hparams, predict_baseline_uncertainty
from mood.constants import RESULTS_DIR, BATCH_SIZE
from mood.model import MOOD_ALGORITHMS, needs_domain_representation, is_domain_generalization
from mood.model.base import Ensemble
from mood.train import train_baseline_model, train
from mood.criteria import get_mood_criteria
from mood.dataset import (
    load_data_from_tdc,
    SimpleMolecularDataset,
    MOOD_REGR_DATASETS,
    domain_based_inference_collate,
)
from mood.distance import get_distance_metric
from mood.metrics import Metric
from mood.representations import featurize
from mood.preprocessing import DEFAULT_PREPROCESSING
from mood.splitter import get_mood_splitters, MOODSplitter
from mood.utils import load_distances_for_downstream_application
from mood.rct import get_experimental_configurations


def run_study(metric, algorithm, n_startup_trials, n_trials, trial_fn, seed):
    """Endpoint for running an Optuna study"""

    direction = "maximize" if metric.mode == "max" else "minimize"
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials)

    study = optuna.create_study(direction=direction, sampler=sampler)

    # ValueError: array must not contain infs or NaNs
    # LinAlgError: N-th leading minor of the array is not positive definite
    # LinAlgError: The kernel is not returning a positive definite matrix
    if algorithm == "GP":
        catch = (np.linalg.LinAlgError, ValueError)
    else:
        catch = ()

    study.optimize(trial_fn, n_trials=n_trials, catch=catch)
    return study


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
    n_trials: int = 50,
    n_startup_trials: int = 10,
):
    """
    This hyper-parameter search loop is used to train baseline models for the MOOD specification.
    All baselines are from scikit-learn.

    NOTE: This could be merged with the more elaborate tuning loop we wrote later
      However, for the sake of reproducibility, I wanted to keep this code intact.
      This way, the exact code used to generate results is still easily accessible
      in the code base
    """

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
            calibrate=True,
        )
        y_pred = model.predict(X_test)
        score = metric(y_test, y_pred)
        return score

    study = run_study(
        metric=metric,
        algorithm=name,
        n_startup_trials=n_startup_trials,
        n_trials=n_trials,
        trial_fn=run_trial,
        seed=global_seed,
    )
    return study


def rct_dataset_setup(dataset, train_indices, val_indices, test_dataset, is_regression):
    """Sets up the dataset. Specifically, splits the dataset and standardizes the targets for regression tasks"""

    train_dataset = dataset.filter_by_indices(train_indices)
    val_dataset = dataset.filter_by_indices(val_indices)

    scaler = None

    # Z-standardization of the targets
    if is_regression:
        scaler = StandardScaler()
        train_dataset.y = scaler.fit_transform(train_dataset.y)
        val_dataset.y = scaler.transform(val_dataset.y)
        test_dataset.y = scaler.transform(test_dataset.y)

    return train_dataset, val_dataset, scaler


def rct_predict_step(model, dataset):
    """Get the predictions and uncertainty estimates from either a scikit-learn model or torch model"""
    if isinstance(model, BaseEstimator):
        y_pred = model.predict(dataset.X).reshape(-1, 1)
        uncertainty = predict_baseline_uncertainty(model, dataset.X)
    elif isinstance(model, Ensemble):
        collate_fn = domain_based_inference_collate if is_domain_generalization(model.models[0]) else None
        dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE)
        y_pred = model.predict(dataloader)
        uncertainty = model.predict_uncertainty(dataloader)
    else:
        raise NotImplementedError
    return y_pred, uncertainty


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
    num_repeated_splits: int = 3,
    num_trials: int = 50,
    num_startup_trials: int = 10,
):
    """
    This hyper-parameter search loop is used to benchmark different tools to improve generalization in
    the MOOD investigation. It combines training scikit-learn and pytorch (lightning) models.
    """

    rng = np.random.default_rng(global_seed)
    seeds = rng.integers(0, 2**16, num_trials)

    def run_trial(trial: optuna.Trial):

        random_state = seeds[trial.number].item()
        trial.set_user_attr("trial_seed", random_state)

        splitters = get_mood_splitters(train_val_dataset.smiles, num_repeated_splits, random_state, n_jobs=-1)
        train_val_splitter = splitters[train_val_split]

        for split_idx, (train_ind, val_ind) in enumerate(train_val_splitter.split(train_val_dataset.X)):

            train_dataset, val_dataset, scaler = rct_dataset_setup(
                train_val_dataset, train_ind, val_ind, test_dataset, is_regression
            )

            if algorithm in MOOD_ALGORITHMS:
                params = MOOD_ALGORITHMS[algorithm].suggest_params(trial)
            else:
                params = suggest_baseline_hparams(algorithm, is_regression, trial)

            model = train(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                algorithm=algorithm,
                is_regression=is_regression,
                params=params,
                seed=random_state,
                calibrate=False,
                # NOTE: If we do not select models based on uncertainty,
                #  we don't train an ensemble to reduce computational cost
                ensemble_size=5 if criterion.needs_uncertainty else 1
            )

            val_y_pred, val_uncertainty = rct_predict_step(model, val_dataset)
            val_prf_score = performance_metric(val_dataset.y, val_y_pred)

            test_y_pred, _ = rct_predict_step(model, test_dataset)
            test_prf_score = performance_metric(test_dataset.y, test_y_pred)

            # We save the val and test performance for each trial to analyze the success
            # of the model selection procedure (gap between best and selected model)
            # NOTE: Ideally we would do this for calibration too, but that was too computationally expensive
            trial.set_user_attr(f"val_performance_{split_idx}", val_prf_score)
            trial.set_user_attr(f"test_performance_{split_idx}", test_prf_score)

            criterion.update(val_y_pred, val_uncertainty, train_dataset, val_dataset)

        return criterion.critique()

    criterion = get_mood_criteria(performance_metric, calibration_metric)[criterion_name]

    study = run_study(
        metric=criterion,
        algorithm=algorithm,
        n_startup_trials=num_startup_trials,
        n_trials=num_trials,
        trial_fn=run_trial,
        seed=global_seed,
    )

    return study


def tune_cmd(
    dataset,
    algorithm,
    representation,
    train_val_split,
    criterion,
    seed: int = 0,
    use_cache: bool = False,
    base_save_dir: str = RESULTS_DIR,
    sub_save_dir: Optional[str] = None,
    overwrite: bool = False,
):
    """
    The MOOD tuning loop: Runs a hyper-parameter search.

    Prescribes a train-test split based on the MOOD specification and runs a hyper-parameter search
    for the training set.
    """

    if sub_save_dir is None:
        sub_save_dir = datetime.now().strftime("%Y%m%d")

    csv_out_dir = dm.fs.join(base_save_dir, "dataframes", "RCT", sub_save_dir)
    csv_fname = f"rct_study_{dataset}_{algorithm}_{representation}_{train_val_split}_{criterion}_{seed}.csv"
    csv_path = dm.fs.join(csv_out_dir, csv_fname)
    dm.fs.mkdir(csv_out_dir, exist_ok=True)

    yaml_out_dir = dm.fs.join(base_save_dir, "YAML", "RCT", sub_save_dir)
    yaml_fname = f"rct_selected_model_{dataset}_{algorithm}_{representation}_{train_val_split}_{criterion}_{seed}.yaml"
    yaml_path = dm.fs.join(yaml_out_dir, yaml_fname)
    dm.fs.mkdir(yaml_out_dir, exist_ok=True)

    if not overwrite and dm.fs.exists(yaml_path) and dm.fs.exists(csv_path):
        logger.info(f"Both the files already exists and overwrite=False. Skipping!")
        return

    # Load and preprocess the data
    smiles, y = load_data_from_tdc(dataset, disable_logs=True)
    X, mask = featurize(smiles, representation, DEFAULT_PREPROCESSING[representation], disable_logs=True)
    X = X.astype(np.float32)
    smiles = smiles[mask]
    y = y[mask]

    is_regression = dataset in MOOD_REGR_DATASETS
    if is_regression:
        y = y.reshape(-1, 1)

    # Prescribe a train-test split
    distances_vs = load_distances_for_downstream_application(
        "virtual_screening", representation, dataset, update_cache=not use_cache
    )
    distances_op = load_distances_for_downstream_application(
        "optimization", representation, dataset, update_cache=not use_cache
    )

    distance_metric = get_distance_metric(X)
    splitters = get_mood_splitters(smiles, 5, seed, n_jobs=-1)
    train_test_splitter = MOODSplitter(
        splitters, np.concatenate((distances_vs, distances_op)), distance_metric, k=5
    )
    train_test_splitter.fit(X)

    # Split the data using the prescribed split
    trainval, test = next(train_test_splitter.split(X, y))
    train_val_dataset = SimpleMolecularDataset(smiles[trainval], X[trainval], y[trainval])
    test_dataset = SimpleMolecularDataset(smiles[test], X[test], y[test])

    if needs_domain_representation(algorithm):
        train_val_dataset.compute_domain_representations()
        test_dataset.compute_domain_representations()

    # Get metrics for this dataset
    performance_metric = Metric.get_default_performance_metric(dataset)
    calibration_metric = Metric.get_default_calibration_metric(dataset)

    # Run the hyper-parameter search
    study = rct_tuning_loop(
        train_val_dataset=train_val_dataset,
        test_dataset=test_dataset,
        algorithm=algorithm,
        train_val_split=train_val_split,
        criterion_name=criterion,
        performance_metric=performance_metric,
        calibration_metric=calibration_metric,
        is_regression=is_regression,
        global_seed=seed,
    )

    # Train the best model found again, but this time as an ensemble
    # to evaluate the test performance and calibration

    splitters = get_mood_splitters(train_val_dataset.smiles, 1, seed, n_jobs=-1)
    train_val_splitter = splitters[train_val_split]
    train_ind, val_ind = next(train_val_splitter.split(train_val_dataset.X))

    train_dataset, val_dataset, scaler = rct_dataset_setup(
        train_val_dataset, train_ind, val_ind, test_dataset, is_regression
    )
    model = train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        algorithm=algorithm,
        is_regression=is_regression,
        params=study.best_params,
        seed=seed,
        calibrate=False,
        ensemble_size=5
    )

    test_y_pred, test_uncertainties = rct_predict_step(model, test_dataset)
    test_prf_score = performance_metric(test_dataset.y, test_y_pred)
    test_cal_score = calibration_metric(test_dataset.y, test_y_pred, test_uncertainties)

    val_y_pred, val_uncertainties = rct_predict_step(model, val_dataset)
    val_prf_score = performance_metric(val_dataset.y, val_y_pred)
    val_cal_score = calibration_metric(val_dataset.y, val_y_pred, val_uncertainties)

    # Save the full trial results as a CSV
    logger.info(f"Saving the full study data to {csv_path}")
    df = study.trials_dataframe()
    df["dataset"] = dataset
    df["algorithm"] = algorithm
    df["representation"] = representation
    df["train-val split"] = train_val_split
    df["criterion"] = criterion
    df["seed"] = seed
    df.to_csv(csv_path)

    # Save the most important information as YAML (higher precision)
    data = {
        "hparams": study.best_params,
        "criterion_final": study.best_value,
        "test_performance_final": test_prf_score,
        "val_performance_final": val_prf_score,
        "test_calibration_final": test_cal_score,
        "val_calibration_final": val_cal_score,
        "dataset": dataset,
        "algorithm": algorithm,
        "representation": representation,
        "train_val_split": train_val_split,
        "criterion": criterion,
        "seed": seed,
        **study.best_trial.user_attrs
    }
    logger.info(f"Saving the data of the best model to {yaml_path}")
    with fsspec.open(yaml_path, "w") as fd:
        yaml.dump(data, fd)


def rct_cmd(
    dataset: str,
    index: int,
    base_save_dir: str = RESULTS_DIR,
    sub_save_dir: Optional[str] = None,
    overwrite: bool = False,
):
    """
    Entrypoint for the benchmarking study in the MOOD Investigation.

    Deterministically samples one of the unordered set of experimental configurations in the RCT.
    And runs the tuning loop for that experimental configuration.

    Here an experimental configuration consists of an algorithm, representation, train-val split,
    model selection criterion and seed.
    """

    configs = get_experimental_configurations(dataset)
    logger.info(f"Sampled configuration #{index} / {len(configs)} for {dataset}: {configs[index]}")
    algorithm, representation, train_val_split, criterion, seed = configs[index]

    tune_cmd(
        dataset=dataset,
        algorithm=algorithm,
        representation=representation,
        train_val_split=train_val_split,
        criterion=criterion,
        seed=seed,
        base_save_dir=base_save_dir,
        sub_save_dir=sub_save_dir,
        overwrite=overwrite,
    )
