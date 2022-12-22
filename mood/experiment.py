import numpy as np
import optuna

from mood.baselines import suggest_baseline_hparams, train_model, predict_uncertainty
from mood.criteria import get_mood_criteria
from mood.dataset import load_data_from_tdc, SimpleMolecularDataset, MOOD_REGR_DATASETS
from mood.distance import get_distance_metric
from mood.metrics import Metric
from mood.representations import featurize
from mood.preprocessing import DEFAULT_PREPROCESSING
from mood.splitter import get_mood_splitters, MOODSplitter
from mood.utils import load_distances_for_downstream_application


def rct_tuning_loop(
    smiles_trainval,
    X_trainval,
    X_test,
    y_trainval,
    y_test,
    algorithm: str,
    criterion: str,
    train_val_split: str,
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

        train_val_splitter = get_mood_splitters(
            smiles_trainval, num_repeated_splits, random_state, n_jobs=-1
        )[train_val_split]

        for train_ind, val_ind in train_val_splitter.split(X_trainval):

            train_dataset = SimpleMolecularDataset(
                smiles_trainval[train_ind], X_trainval[train_ind], y_trainval[train_ind]
            )
            val_dataset = SimpleMolecularDataset(
                smiles_trainval[val_ind], X_trainval[val_ind], y_trainval[val_ind]
            )

            random_state = global_seed + trial.number
            params = suggest_baseline_hparams(algorithm, is_regression, trial)
            model = train_model(
                train_dataset.X,
                train_dataset.y,
                algorithm,
                is_regression,
                params,
                random_state,
                for_uncertainty_estimation=True,
                ensemble_size=10,
            )
            y_pred = model.predict(val_dataset.X)
            uncertainty = predict_uncertainty(model, val_dataset.X)
            criterion.update(y_pred, uncertainty, train_dataset, val_dataset)

        return criterion.critique()

    criterion = get_mood_criteria(performance_metric, calibration_metric)[criterion]

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
    model = train_model(
        X_trainval,
        y_trainval,
        algorithm,
        is_regression,
        study.best_params,
        global_seed + study.best_trial.number,
        for_uncertainty_estimation=True,
        ensemble_size=10,
    )

    y_pred = model.predict(X_test)
    uncertainty = predict_uncertainty(model, X_test)

    cal_score = calibration_metric(y_test, y_pred, uncertainty)
    prf_score = calibration_metric(y_test, y_pred, uncertainty)
    return study, cal_score, prf_score


def train(dataset, representation, algorithm, criterion, train_val_split, seed: int = 0, use_cache: bool = False):

    smiles, y = load_data_from_tdc(dataset, disable_logs=True)
    X, mask = featurize(smiles, representation, DEFAULT_PREPROCESSING[representation], disable_logs=True)
    smiles = smiles[mask]
    y = y[mask]

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

    study, cal_score, prf_score = rct_tuning_loop(
        smiles_trainval=smiles[trainval],
        X_trainval=X[trainval],
        y_trainval=y[trainval],
        X_test=X[test],
        y_test=y[test],
        algorithm=algorithm,
        criterion=criterion,
        train_val_split=train_val_split,
        performance_metric=Metric.get_default_performance_metric(dataset),
        calibration_metric=Metric.get_default_calibration_metric(dataset),
        is_regression=dataset in MOOD_REGR_DATASETS,
        global_seed=seed,
    )

    print(cal_score, prf_score)


if __name__ == "__main__":
    train("Clearance", "MACCS", "RF", "Calibration x Performance", "Random", 0)
