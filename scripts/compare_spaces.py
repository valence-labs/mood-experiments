import typer
import fsspec
import datamol as dm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from loguru import logger
from typing import List, Optional

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from sklearn.gaussian_process.kernels import PairwiseKernel, Sum, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier

from mood.dataset import dataset_iterator, MOOD_REGR_DATASETS
from mood.model_space import ModelSpaceTransformer
from mood.preprocessing import DEFAULT_PREPROCESSING
from mood.distance import compute_knn_distance
from mood.visualize import plot_distance_distributions
from mood.representations import representation_iterator, featurize
from mood.constants import RESULTS_DIR
from mood.utils import (
    load_representation_for_downstream_application,
    save_figure_with_fsspec,
    get_outlier_bounds,
)
from mood.train import train_baseline_model


def train_gp(X, y, is_regression):
    alpha = 1e-10
    for i in range(10):
        try:
            if is_regression:
                kernel = PairwiseKernel(metric="linear")
                model = GaussianProcessRegressor(kernel, alpha=alpha, random_state=0)
            else:
                kernel = Sum(PairwiseKernel(metric="linear"), WhiteKernel(noise_level=alpha))
                model = GaussianProcessClassifier(kernel, random_state=0)
            model.fit(X, y)
            return model
        except (np.linalg.LinAlgError, ValueError):
            # ValueError: array must not contain infs or NaNs
            # LinAlgError: N-th leading minor of the array is not positive definite
            # LinAlgError: The kernel is not returning a positive definite matrix
            alpha = alpha * 10
    return None


def get_model_space_distances(model, train, queries):

    embedding_size = int(round(train.shape[1] * 0.25))
    trans = ModelSpaceTransformer(model, embedding_size)

    model_space_train = trans(train)
    queries = [trans(q) for q in queries]

    distances = compute_knn_distance(model_space_train, queries, n_jobs=-1)
    return distances


def compute_correlations(input_spaces, model_spaces, labels):

    lower, upper = get_outlier_bounds(np.concatenate(input_spaces), factor=3.0)
    input_masks = [(X >= lower) & (X <= upper) for X in input_spaces]

    lower, upper = get_outlier_bounds(np.concatenate(model_spaces), factor=3.0)
    model_masks = [(X >= lower) & (X <= upper) for X in model_spaces]

    masks = [mask1 & mask2 for mask1, mask2 in zip(input_masks, model_masks)]
    input_spaces = [d[mask] for d, mask in zip(input_spaces, masks)]
    model_spaces = [d[mask] for d, mask in zip(model_spaces, masks)]

    df = pd.DataFrame()
    for input_space, model_space, label in zip(input_spaces, model_spaces, labels):

        df_ = pd.DataFrame(
            {
                "pearson": pearsonr(input_space, model_space)[0],
                "spearman": spearmanr(input_space, model_space)[0],
                "r_squared": r2_score(input_space, model_space),
            },
            index=[0],
        )
        df = pd.concat((df, df_), ignore_index=True)

    return df


def cli(
    base_save_dir: str = RESULTS_DIR,
    sub_save_dir: Optional[str] = None,
    overwrite: bool = False,
    skip_representation: Optional[List[str]] = None,
    skip_dataset: Optional[List[str]] = None,
    batch_size: int = 16,
):

    if sub_save_dir is None:
        sub_save_dir = datetime.now().strftime("%Y%m%d")

    fig_save_dir = dm.fs.join(base_save_dir, "figures", "compare_spaces", sub_save_dir)
    dm.fs.mkdir(fig_save_dir, exist_ok=True)
    logger.info(f"Saving figures to {fig_save_dir}")

    np_save_dir = dm.fs.join(base_save_dir, "numpy", "compare_spaces", sub_save_dir)
    dm.fs.mkdir(np_save_dir, exist_ok=True)
    logger.info(f"Saving NumPy arrays to {np_save_dir}")

    df_save_dir = dm.fs.join(base_save_dir, "dataframes", "compare_spaces", sub_save_dir)
    dm.fs.mkdir(df_save_dir, exist_ok=True)
    corr_path = dm.fs.join(df_save_dir, "correlations.csv")
    if dm.fs.exists(df_save_dir) and not overwrite:
        raise typer.BadParameter(f"{corr_path} already exists. Specify --overwrite or --base-save-dir.")
    logger.info(f"Saving correlation dataframe to {corr_path}")

    df_corr = pd.DataFrame()

    dataset_it = dataset_iterator(blacklist=skip_dataset)

    for dataset, (smiles, y) in dataset_it:

        representation_it = representation_iterator(
            smiles,
            n_jobs=-1,
            progress=True,
            blacklist=skip_representation,
            standardize_fn=DEFAULT_PREPROCESSING,
            batch_size=batch_size,
        )

        for representation, (X, mask) in representation_it:

            y_repr = y[mask]

            virtual_screening = load_representation_for_downstream_application(
                "virtual_screening", representation, update_cache=True
            )
            optimization = load_representation_for_downstream_application(
                "optimization", representation, update_cache=True
            )

            is_regression = dataset in MOOD_REGR_DATASETS
            mlp_model = train_baseline_model(X, y_repr, "MLP", is_regression)
            rf_model = train_baseline_model(X, y_repr, "RF", is_regression)
            # We use a custom train function for GPs to include a retry system
            gp_model = train_gp(X, y_repr, is_regression)

            # Distances in input spaces
            input_distances = compute_knn_distance(X, [X, optimization, virtual_screening], n_jobs=-1)
            labels = ["Train", "Optimization", "Virtual Screening"]

            for dist, label in zip(input_distances, labels):
                path = dm.fs.join(np_save_dir, f"{dataset}_{representation}_{label}_input_space.npy")
                if dm.fs.exists(df_save_dir) and not overwrite:
                    raise RuntimeError(f"{path} already exists. Specify --overwrite or --base-save-dir.")
                with fsspec.open(path, "wb") as fd:
                    np.save(fd, dist)

            ax = plot_distance_distributions(input_distances, labels=labels)
            ax.set_title(f"Input space ({representation}, {dataset})")
            save_figure_with_fsspec(
                dm.fs.join(fig_save_dir, f"{dataset}_{representation}_input_space.png"), exist_ok=overwrite
            )
            plt.close()

            # Distances in different model spaces
            for name, model in {"MLP": mlp_model, "RF": rf_model, "GP": gp_model}.items():

                if model is None:
                    logger.warning(f"Failed to train a {name} model for {dataset} on {representation}")
                    continue

                model_distances = get_model_space_distances(model, X, [X, optimization, virtual_screening])
                for dist, label in zip(model_distances, labels):
                    path = dm.fs.join(np_save_dir, f"{dataset}_{representation}_{label}_{name}_space.npy")
                    if dm.fs.exists(df_save_dir) and not overwrite:
                        raise RuntimeError(f"{path} already exists. Specify --overwrite or --base-save-dir.")
                    with fsspec.open(path, "wb") as fd:
                        np.save(fd, dist)

                ax = plot_distance_distributions(model_distances, labels=labels)
                ax.set_title(f"{name} space ({representation}, {dataset})")
                save_figure_with_fsspec(
                    dm.fs.join(fig_save_dir, f"{dataset}_{representation}_{name}_space.png"),
                    exist_ok=overwrite,
                )
                plt.close()

                # Compute correlation
                df = compute_correlations(
                    input_distances,
                    model_distances,
                    labels,
                )
                df["model"] = name
                df["dataset"] = dataset
                df["representation"] = representation
                df_corr = pd.concat((df_corr, df), ignore_index=True)

    df_corr.to_csv(corr_path, index=False)
    df_corr.head()
