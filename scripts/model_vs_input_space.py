import typer
import datamol as dm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
from loguru import logger
from typing import List, Optional

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import PairwiseKernel, Sum, WhiteKernel

from mood.dataset import dataset_iterator, load_data_from_tdc, MOOD_REGR_DATASETS, MOOD_CLSF_DATASETS
from mood.model_space import ModelSpaceTransformer
from mood.preprocessing import standardize_smiles
from mood.distance import compute_knn_distance
from mood.visualize import plot_distance_distributions
from mood.representations import representation_iterator, featurize
from mood.constants import DOWNSTREAM_APPS_DATA_DIR
from mood.utils import load_representation_for_downstream_application, save_figure_with_fsspec, get_outlier_bounds


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
                
    
def train_model(X, y, name: str, is_regression: bool):
    
    models = {
        "mlp": {
            "regression":  MLPRegressor(hidden_layer_sizes=(128, 128, X.shape[1],), random_state=0),
            "classification": MLPClassifier(hidden_layer_sizes=(128, 128, X.shape[1],), random_state=0),
        },
        "rf": {
            "regression":  RandomForestRegressor(random_state=0),
            "classification": RandomForestClassifier(random_state=0),
        }
    }
    
    if name == "gp":
        model = train_gp(X, y, is_regression)
    else:
        target_type = "regression" if is_regression else "classification"
        model = models[name][target_type]
        model.fit(X, y)
    return model


def get_model_space_distances(model, train, queries):
    
    embedding_size = int(round(train.shape[1] * 0.25))
    trans = ModelSpaceTransformer(model, embedding_size)

    model_space_train = trans(train)
    queries = [trans(q) for q in queries]

    distances = compute_knn_distance(
        model_space_train, 
        queries, 
        n_jobs=-1
    )
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
        
        df_ = pd.DataFrame({
            "pearson": pearsonr(input_space, model_space)[0],
            "spearman": spearmanr(input_space, model_space)[0],
            "r_squared": r2_score(input_space, model_space),
        }, index=[0])
        df = pd.concat((df, df_), ignore_index=True)
        
    return df


def cli(
    base_save_dir: str = "gs://experiments-output/mood-v2/results/",
    overwrite: bool = False,
    representation: Optional[List[str]] = None,
    dataset: Optional[List[str]] = None,
):
        
    today = datetime.now().strftime("%Y%m%d")
    fig_save_dir = dm.fs.join(base_save_dir, "figures", f"{today}_NB01")
    dm.fs.mkdir(fig_save_dir, exist_ok=True)
    logger.info(f"Saving figures to {fig_save_dir}")
    
    corr_save_dir = dm.fs.join(base_save_dir, "dataframes", f"{today}_NB01")
    dm.fs.mkdir(corr_save_dir, exist_ok=True)
    corr_path = dm.fs.join(corr_save_dir, "correlations.csv")
    if dm.fs.exists(corr_path) and not overwrite:
        raise typer.BadParameter(f"{corr_path} already exists. Specify --overwrite or --base-save-dir.")
    logger.info(f"Saving correlation dataframe to {corr_path}")

        
    df_corr = pd.DataFrame()
    
    if len(dataset) == 0:
        dataset = None
    if len(representation) == 0:
        representation = None

    dataset_it = dataset_iterator(standardize_smiles, progress=True, whitelist=dataset)
    
    for dataset, (smiles, y) in dataset_it:
        representation_it = representation_iterator(
            smiles, n_jobs=-1, progress=True, whitelist=representation
        )

        for representation, (X, mask) in representation_it:

            y_repr = y[mask]

            virtual_screening = load_representation_for_downstream_application("virtual_screening", representation)
            optimization = load_representation_for_downstream_application("optimization", representation)      

            is_regression = dataset in MOOD_REGR_DATASETS
            mlp_model = train_model(X, y_repr, "mlp", is_regression)
            rf_model = train_model(X, y_repr, "rf", is_regression)
            gp_model = train_model(X, y_repr, "gp", is_regression)

            # Distances in input spaces
            input_distances = compute_knn_distance(X, [X, optimization, virtual_screening], n_jobs=-1)

            labels = ["Train", "Optimization", "Virtual Screening"]
            
            ax = plot_distance_distributions(input_distances, labels=labels)
            ax.set_title(f"Input space ({representation}, {dataset})")
            save_figure_with_fsspec(dm.fs.join(fig_save_dir, f"{dataset}_{representation}_input_space.png"), exist_ok=overwrite)
            plt.close()

            # Distances in different model spaces
            for name, model in {"MLP": mlp_model, "RF": rf_model, "GP": gp_model}.items():

                if model is None:
                    logger.warning(f"Failed to train a {name} model for {dataset} on {representation}")
                    continue 

                model_distances = get_model_space_distances(model, X, [X, optimization, virtual_screening])
            
                ax = plot_distance_distributions(model_distances, labels=labels)
                ax.set_title(f"{name} space ({representation}, {dataset})")
                save_figure_with_fsspec(
                    dm.fs.join(fig_save_dir, f"{dataset}_{representation}_{name}_space.png"), 
                    exist_ok=overwrite
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
