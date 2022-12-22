import numpy as np
import matplotlib.pyplot as plt

from typing import Optional
from loguru import logger

from mood.distance import get_distance_metric
from mood.representations import featurize
from mood.dataset import load_data_from_tdc
from mood.utils import load_distances_for_downstream_application
from mood.splitter import MOODSplitter, get_mood_splitters
from mood.preprocessing import DEFAULT_PREPROCESSING


def cli(
    dataset: str,
    representation: str,
    n_splits: int = 5,
    use_cache: bool = True,
    seed: Optional[int] = None,
):

    logger.info(f"Loading precomputed distances for virtual screening")
    distances_vs = load_distances_for_downstream_application(
        "virtual_screening", representation, dataset, update_cache=not use_cache
    )

    logger.info(f"Loading precomputed distances for optimization")
    distances_op = load_distances_for_downstream_application(
        "optimization", representation, dataset, update_cache=not use_cache
    )

    smiles, y = load_data_from_tdc(dataset)
    standardize_fn = DEFAULT_PREPROCESSING[representation]
    X, mask = featurize(smiles, representation, standardize_fn, disable_logs=True)
    y = y[mask]

    metric = get_distance_metric(X)
    if metric == "jaccard":
        X = X.astype(bool)

    splitters = get_mood_splitters(smiles[mask], n_splits, seed)
    splitter = MOODSplitter(splitters, np.concatenate((distances_vs, distances_op)), metric, k=5)
    ax = splitter.fit(X, y, plot=True, progress=False)
    plt.show()
