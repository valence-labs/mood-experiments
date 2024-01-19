import numpy as np

from typing import Optional, Union, List

from sklearn.neighbors import NearestNeighbors


def get_distance_metric(example):
    """Get the appropriate distance metric given an exemplary datapoint"""

    # By default we use the Euclidean distance
    metric = "euclidean"

    # For binary vectors we use jaccard
    if ((example == 0) | (example == 1)).all():
        metric = "jaccard"

    return metric


def compute_knn_distance(
    X: np.ndarray,
    Y: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    metric: Optional[str] = None,
    k: int = 5,
    n_jobs: Optional[int] = None,
    return_indices: bool = False,
):
    """
    Computes the mean k-Nearest Neighbors distance
    between a set of database embeddings and a set of query embeddings

    Args:
        X: The set of samples that form kNN candidates
        Y: The samples for which to find the kNN for. If None, will find kNN for `database`
        metric: The pairwise distance metric to define the neighborhood
        k: The number of neighbors to find
        n_jobs: Controls the parallelization
        return_indices: Whether to return the indices of the NNs as well
    """

    if metric is None:
        metric = get_distance_metric(X[0])

    knn = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=n_jobs)
    knn.fit(X)

    if not isinstance(Y, list):
        Y = [Y]

    distances, indices = [], []
    for queries in Y:
        if np.array_equal(X, queries):
            # Use k + 1 and filter out the first
            # because the sample will always be its own neighbor
            dist, ind = knn.kneighbors(queries, n_neighbors=k + 1)
            dist, ind = dist[:, 1:], ind[:, 1:]
        else:
            dist, ind = knn.kneighbors(queries, n_neighbors=k)

        distances.append(dist)
        indices.append(ind)

    # The distance from the query molecule to its NNs is the mean of all pairwise distances
    distances = [np.mean(dist, axis=1) for dist in distances]

    if len(distances) == 1:
        assert len(indices) == 1
        distances = distances[0]
        indices = indices[0]

    if return_indices:
        return distances, indices
    return distances
