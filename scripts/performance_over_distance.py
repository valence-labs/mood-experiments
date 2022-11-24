import tqdm

import pandas as pd
import numpy as np
import datamol as dm
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
from loguru import logger
from sklearn.model_selection import train_test_split

from mood.constants import DOWNSTREAM_RESULTS_DIR
from mood.dataset import load_data_from_tdc, TDC_TO_METRIC, MOOD_REGR_DATASETS
from mood.representations import featurize
from mood.baselines import tune_model, train_model
from mood.utils import bin_with_overlap, load_distances_for_downstream_application, get_outlier_bounds
from mood.distance import compute_knn_distance
from mood.metrics import compute_bootstrapped_metric, compute_metric


def cli(
    baseline_algorithm: str,
    representation: str,
    dataset: str,
    n_seeds: int = 5,
    n_trials: int = 50,
    n_startup_trials: int = 10,
    base_save_dir: str = DOWNSTREAM_RESULTS_DIR,
    overwrite: bool = False,
):
    
    dir_name = f"{datetime.now().strftime('%Y%m%d')}_NB02"
    out_dir = dm.fs.join(base_save_dir, "dataframes", dir_name)
    dm.fs.mkdir(out_dir, exist_ok=True)
    
    smiles, y = load_data_from_tdc(dataset)
    X, mask = featurize(smiles, representation)
    y = y[mask]
    
    metric = TDC_TO_METRIC[dataset]
    is_regression = dataset in MOOD_REGR_DATASETS
    
    dist_train = []
    dist_val = []
    y_pred = []
    y_true = []
    
    for seed in range(n_seeds):
        
        file_name = f"trials_{dataset}_{baseline_algorithm}_{representation}_{seed}.csv"
        out_path = dm.fs.join(out_dir, file_name)
        
        if dm.fs.exists(out_path) and not overwrite:
            raise RuntimeError(f"{out_path} already exists!")
            
        # Randomly split the dataset
        # This ensures that the distribution of distances from val to train is relatively uniform
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)
                
        study = tune_model(
            X_train=X_train, 
            X_test=X_val, 
            y_train=y_train, 
            y_test=y_val, 
            name=baseline_algorithm, 
            is_regression=is_regression, 
            metric=metric, 
            global_seed=seed,
            n_trials=n_trials,
            n_startup_trials=n_startup_trials,
        )

        model = train_model(
            X_train, 
            y_train, 
            baseline_algorithm, 
            is_regression, 
            study.best_params, 
            seed + study.best_trial.number
        )
        
        logger.info(f"Saving the trials dataframe to {out_path}")
        study.trials_dataframe().to_csv(out_path)

        y_pred_ = model.predict(X_val)
        y_pred.append(y_pred_)
        y_true.append(y_val)
        
        dist_train_, dist_val_ = compute_knn_distance(X_train, [X_train, X_val])
        dist_train.append(dist_train_)
        dist_val.append(dist_val_)
    
    dist_val = np.concatenate(dist_val)
    dist_train = np.concatenate(dist_train)
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    
    dist_scr = load_distances_for_downstream_application("virtual_screening", representation, dataset)
    dist_opt = load_distances_for_downstream_application("optimization", representation, dataset)
    dist_app = np.concatenate((dist_opt, dist_scr))
    
    lower, upper = np.quantile(dist_train, 0.025), np.quantile(dist_train, 0.975)
    mask = np.logical_and(dist_val >= lower, dist_val <= upper)
    score_iid = compute_metric(y_pred[mask], y_true[mask], metric)
    logger.info(f"Found an IID {metric} score of {score_iid:.3f}")

    lower, upper = np.quantile(dist_app, 0.025), np.quantile(dist_app, 0.975)
    mask = np.logical_and(dist_val >= lower, dist_val <= upper)
    score_ood = compute_metric(y_pred[mask], y_true[mask], metric)
    logger.info(f"Found an OOD {metric} score of {score_ood:.3f}")
    
    file_name = f"gap_{dataset}_{baseline_algorithm}_{representation}.csv"
    out_path = dm.fs.join(out_dir, file_name)
        
    if dm.fs.exists(out_path) and not overwrite:
        raise RuntimeError(f"{out_path} already exists!")
    
    # Saving this as a CSV might be a bit wasteful, 
    # but I am not sure what makes more sense
    logger.info(f"Saving the IID/OOD gap data to {out_path}")
    study.trials_dataframe().to_csv(out_path)
        
    df = pd.DataFrame({
        "dataset": dataset,
        "algorithm": baseline_algorithm,
        "representation": representation,
        "iid_score": score_iid,
        "ood_score": score_ood,
        "metric": metric,
    }, index=[0]).to_csv(out_path, index=False)
