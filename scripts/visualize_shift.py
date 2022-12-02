import matplotlib.pyplot as plt

from typing import Optional
from loguru import logger

from mood.dataset import load_data_from_tdc, MOOD_REGR_DATASETS
from mood.representations import featurize
from mood.preprocessing import DEFAULT_PREPROCESSING
from mood.visualize import plot_distance_distributions
from mood.distance import compute_knn_distance
from mood.utils import load_representation_for_downstream_application
from mood.baselines import train_model
from mood.transformer import ModelSpaceTransformer


def cli(
    dataset: str, 
    representation: str, 
    model_space: Optional[str] = None,
):
    smiles, y = load_data_from_tdc(dataset)
    standardize_fn = DEFAULT_PREPROCESSING[representation]
    X, mask = featurize(smiles, representation, standardize_fn, disable_logs=True)
    y = y[mask]
    
    logger.info(f"Loading precomputed representations for virtual screening")
    virtual_screening = load_representation_for_downstream_application("virtual_screening", representation)
    
    logger.info(f"Loading precomputed representations for optimization")
    optimization = load_representation_for_downstream_application("optimization", representation)      
    
    if model_space is not None:
        
        logger.info(f"Computing distance in the {model_space} model space")
        is_regression = dataset in MOOD_REGR_DATASETS
        model = train_model(X, y, model_space, is_regression)
        embedding_size = int(round(X.shape[1] * 0.25))
        trans = ModelSpaceTransformer(model, embedding_size)

        X = trans(X)
        virtual_screening = trans(virtual_screening)
        optimization = trans(optimization)
    
    logger.info("Computing the k-NN distance")
    distances = compute_knn_distance(X, [X, optimization, virtual_screening], n_jobs=-1)
    
    logger.info("Plotting the results")
    labels = ["Train", "Optimization", "Virtual Screening"]
    ax = plot_distance_distributions(distances, labels=labels)
    plt.show()
