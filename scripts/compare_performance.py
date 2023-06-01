import yaml
import fsspec

import pandas as pd
import numpy as np
import datamol as dm

from typing import Optional
from datetime import datetime
from loguru import logger
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mood.constants import RESULTS_DIR
from mood.dataset import load_data_from_tdc, MOOD_REGR_DATASETS
from mood.metrics import Metric, compute_bootstrapped_metric
from mood.representations import featurize
from mood.baselines import predict_baseline_uncertainty
from mood.train import train_baseline_model
from mood.experiment import basic_tuning_loop
from mood.utils import bin_with_overlap, load_distances_for_downstream_application
from mood.distance import compute_knn_distance
from mood.preprocessing import DEFAULT_PREPROCESSING


def cli(
    baseline_algorithm: str,
    representation: str,
    dataset: str,
    n_seeds: int = 5,
    n_trials: int = 50,
    n_startup_trials: int = 10,
    base_save_dir: str = RESULTS_DIR,
    sub_save_dir: Optional[str] = None,
    overwrite: bool = False,
):
    if sub_save_dir is None:
        sub_save_dir = datetime.now().strftime("%Y%m%d")
    out_dir = dm.fs.join(base_save_dir, "dataframes", "compare_performance", sub_save_dir)
    dm.fs.mkdir(out_dir, exist_ok=True)

    # Load the dataset
    smiles, y = load_data_from_tdc(dataset)
    X, mask = featurize(
        smiles,
        representation,
        standardize_fn=DEFAULT_PREPROCESSING[representation],
        disable_logs=True,
    )
    y = y[mask]
    is_regression = dataset in MOOD_REGR_DATASETS

    # Get the metrics
    perf_metric = Metric.get_default_performance_metric(dataset)
    cali_metric = Metric.get_default_calibration_metric(dataset)

    # Generate all data needed for these plots
    dist_train = []
    dist_test = []
    y_pred = []
    y_true = []
    y_uncertainty = []

    for seed in range(n_seeds):
        # Randomly split the dataset
        # This ensures that the distribution of distances from val to train is relatively uniform
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)

        file_name = f"best_hparams_{dataset}_{baseline_algorithm}_{representation}_{seed}.yaml"
        out_path = dm.fs.join(out_dir, file_name)

        if dm.fs.exists(out_path):
            # Load the data of the completed hyper-param study if it already exists
            logger.info(f"Loading the best hyper-params from {out_path}")
            with fsspec.open(out_path) as fd:
                params = yaml.safe_load(fd)

        else:
            # Run a hyper-parameter search
            study = basic_tuning_loop(
                X_train=X_train,
                X_test=X_val,
                y_train=y_train,
                y_test=y_val,
                name=baseline_algorithm,
                is_regression=is_regression,
                metric=perf_metric,
                global_seed=seed,
                n_trials=n_trials,
                n_startup_trials=n_startup_trials,
            )

            params = study.best_params
            random_state = seed + study.best_trial.number
            params["random_state"] = random_state

            logger.info(f"Saving the best hyper-params to {out_path}")
            with fsspec.open(out_path, "w") as fd:
                yaml.dump(params, fd)

            file_name = f"trials_{dataset}_{baseline_algorithm}_{representation}_{seed}.csv"
            out_path = dm.fs.join(out_dir, file_name)

            logger.info(f"Saving the trials dataframe to {out_path}")
            study.trials_dataframe().to_csv(out_path)

        random_state = params.pop("random_state")
        model = train_baseline_model(
            X_train,
            y_train,
            baseline_algorithm,
            is_regression,
            params,
            random_state,
            for_uncertainty_estimation=True,
            ensemble_size=10,
        )

        y_pred_ = model.predict(X_test)
        y_uncertainty_ = predict_baseline_uncertainty(model, X_test)

        y_pred.append(y_pred_)
        y_true.append(y_test)
        y_uncertainty.append(y_uncertainty_)

        dist_train_, dist_test_ = compute_knn_distance(X_train, [X_train, X_test])
        dist_train.append(dist_train_)
        dist_test.append(dist_test_)

    dist_test = np.concatenate(dist_test)
    dist_train = np.concatenate(dist_train)
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    y_uncertainty = np.concatenate(y_uncertainty)

    # Collect the distances of the downstream applications
    dist_scr = load_distances_for_downstream_application(
        "virtual_screening", representation, dataset, update_cache=True
    )
    dist_opt = load_distances_for_downstream_application(
        "optimization", representation, dataset, update_cache=True
    )
    dist_app = np.concatenate((dist_opt, dist_scr))

    # Compute the difference in IID and OOD performance and calibration
    lower, upper = np.quantile(dist_train, 0.025), np.quantile(dist_train, 0.975)
    mask = np.logical_and(dist_test >= lower, dist_test <= upper)
    score_iid = perf_metric(y_true[mask], y_pred[mask])
    calibration_iid = cali_metric(y_true[mask], y_pred[mask], y_uncertainty[mask])
    logger.info(f"Found an IID {perf_metric.name} score of {score_iid:.3f}")
    logger.info(f"Found an IID {cali_metric.name} calibration score of {calibration_iid:.3f}")

    lower, upper = np.quantile(dist_app, 0.025), np.quantile(dist_app, 0.975)
    mask = np.logical_and(dist_test >= lower, dist_test <= upper)
    score_ood = perf_metric(y_true[mask], y_pred[mask])
    calibration_ood = cali_metric(y_true[mask], y_pred[mask], y_uncertainty[mask])
    logger.info(f"Found an OOD {perf_metric.name} score of {score_ood:.3f}")
    logger.info(f"Found an OOD {cali_metric.name} calibration score of {calibration_ood:.3f}")

    file_name = f"gap_{dataset}_{baseline_algorithm}_{representation}.csv"
    out_path = dm.fs.join(out_dir, file_name)
    if dm.fs.exists(out_path) and not overwrite:
        raise RuntimeError(f"{out_path} already exists!")

    # Saving this as a CSV might be a bit wasteful,
    # but it's convenient
    logger.info(f"Saving the IID/OOD gap data to {out_path}")

    pd.DataFrame(
        {
            "dataset": dataset,
            "algorithm": baseline_algorithm,
            "representation": representation,
            "iid_score": [score_iid, calibration_iid],
            "ood_score": [score_ood, calibration_ood],
            "metric": [perf_metric.name, cali_metric.name],
            "type": ["performance", "calibration"],
        }
    ).to_csv(out_path, index=False)

    # Compute the performance over distance
    df = pd.DataFrame()
    for distance, mask in tqdm(bin_with_overlap(dist_test)):
        target = y_true[mask]
        preds = y_pred[mask]
        uncertainty = y_uncertainty[mask]

        n_samples = len(mask)
        if n_samples < 25 or len(np.unique(target)) == 1:
            continue

        perf_mu, perf_std = compute_bootstrapped_metric(
            targets=target, predictions=preds, metric=perf_metric, n_jobs=-1
        )

        cali_mu, cali_std = compute_bootstrapped_metric(
            targets=target, predictions=preds, uncertainties=uncertainty, metric=cali_metric, n_jobs=-1
        )

        df_ = pd.DataFrame(
            {
                "dataset": dataset,
                "algorithm": baseline_algorithm,
                "representation": representation,
                "distance": distance,
                "score_mu": [perf_mu, cali_mu],
                "score_std": [perf_std, cali_std],
                "type": ["performance", "calibration"],
                "metric": [perf_metric.name, cali_metric.name],
                "n_samples": n_samples,
            }
        )
        df = pd.concat((df, df_), ignore_index=True)

    file_name = f"perf_over_distance_{dataset}_{baseline_algorithm}_{representation}.csv"
    out_path = dm.fs.join(out_dir, file_name)
    if dm.fs.exists(out_path) and not overwrite:
        raise RuntimeError(f"{out_path} already exists!")

    logger.info(f"Saving the performance over distance data to {out_path}")
    df.to_csv(out_path, index=False)
