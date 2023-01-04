import tempfile
import uuid
from datetime import datetime
from typing import Optional

import datamol as dm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from loguru import logger

from mood.constants import DOWNSTREAM_APPS_DATA_DIR, CACHE_DIR


def load_representation_for_downstream_application(
    name,
    representation,
    update_cache: bool = False,
    return_compound_ids: bool = False,
):

    suffix = ["representations", name, f"{representation}.parquet"]

    lpath = dm.fs.join(CACHE_DIR, "downstream_applications", *suffix)
    if not dm.fs.exists(lpath) or update_cache:
        rpath = dm.fs.join(DOWNSTREAM_APPS_DATA_DIR, *suffix)
        logger.debug(f"Downloading {rpath} to {lpath}")
        dm.fs.copy_file(rpath, lpath, force=update_cache)
    else:
        logger.debug(f"Using cache at {lpath}")

    data = pd.read_parquet(lpath)

    X = np.stack(data["representation"].values)

    mask = get_mask_for_distances_or_representations(X)

    if not return_compound_ids:
        return X[mask]

    indices = data.iloc[mask]["unique_id"].to_numpy()
    return X[mask], indices


def load_distances_for_downstream_application(
    name,
    representation,
    dataset,
    update_cache: bool = False,
    return_compound_ids: bool = False,
):

    suffix = ["distances", name, dataset, f"{representation}.parquet"]

    lpath = dm.fs.join(CACHE_DIR, "downstream_applications", *suffix)
    if not dm.fs.exists(lpath) or update_cache:
        rpath = dm.fs.join(DOWNSTREAM_APPS_DATA_DIR, *suffix)
        logger.debug(f"Downloading {rpath} to {lpath}")
        dm.fs.copy_file(rpath, lpath, force=update_cache)
    else:
        logger.debug(f"Using cache at {lpath}")

    data = pd.read_parquet(lpath)

    distances = data["distance"].to_numpy()
    mask = get_mask_for_distances_or_representations(distances)

    if not return_compound_ids:
        return distances[mask]

    indices = data.iloc[mask]["unique_id"].to_numpy()
    return distances[mask], indices


def save_figure_with_fsspec(path, exist_ok=False):

    if dm.fs.exists(path) and not exist_ok:
        raise RuntimeError(f"{path} already exists")

    if dm.fs.is_local_path(path):
        plt.savefig(path)
        return

    mapper = dm.fs.get_mapper(path)
    clean_path = path.rstrip(mapper.fs.sep)
    dir_components = str(clean_path).split(mapper.fs.sep)[:-1]
    dir_path = mapper.fs.sep.join(dir_components)

    dm.fs.mkdir(dir_path, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        lpath = dm.fs.join(tmpdir, f"{str(uuid.uuid4())}.png")

        plt.savefig(lpath)
        dm.fs.copy_file(lpath, path, force=exist_ok)


def get_outlier_bounds(X, factor: float = 1.5):

    q1 = np.quantile(X, 0.25)
    q3 = np.quantile(X, 0.75)
    iqr = q3 - q1

    lower = max(np.min(X), q1 - factor * iqr)
    upper = min(np.max(X), q3 + factor * iqr)

    return lower, upper


def bin_with_overlap(data, filter_outliers: bool = True):

    if filter_outliers:
        minimum, maximum = get_outlier_bounds(data)
        window_size = (maximum - minimum) / 10
        yield minimum, np.nonzero(data <= minimum)[0]

    else:
        minimum = np.min(data)
        maximum = np.max(data)
        window_size = (maximum - minimum) / 10

    assert minimum >= 0, "A distance cannot be lower than 0"

    x = minimum
    step_size = window_size / 20
    while x + window_size < maximum:
        yield x + 0.5 * window_size, np.nonzero(np.logical_and(data >= x, data < x + window_size))[0]
        x += step_size

    # Yield the rest data
    yield x + ((maximum - x) / 2.0), np.nonzero(data >= x)[0]


def get_mask_for_distances_or_representations(X):

    # The 1e4 threshold is somewhat arbitrary, but manually chosen to
    # filter out compounds that don't make sense (and are outliers).
    # (e.g. WHIM for [C-]#N.[C-]#N.[C-]#N.[C-]#N.[C-]#N.[Fe+4].[N-]=O)
    # Propagating such high-values would cause issues in downstream
    # functions (e.g. KMeans)
    mask = [
        i
        for i, a in enumerate(X)
        if a is not None and ~np.isnan(a).any() and np.isfinite(a).all() and ~(a > 1e4).any()
    ]
    return mask


class Timer:
    """Context manager for timing operations"""

    def __init__(self, name: Optional[str] = None):
        self.name = name if name is not None else "operation"
        self.start_time = None
        self.end_time = None

    @property
    def duration(self):
        if self.start_time is None:
            raise RuntimeError("Cannot get the duration for an operation that has not started yet")
        if self.end_time is None:
            return datetime.now() - self.start_time
        return self.end_time - self.start_time

    def __enter__(self):
        self.start_time = datetime.now()
        logger.debug(f"Starting {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        logger.info(f"Finished {self.name}. Duration: {self.duration}")
