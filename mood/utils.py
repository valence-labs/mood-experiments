import fsspec
import tempfile
import uuid

import datamol as dm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mood.constants import DOWNSTREAM_APPS_DATA_DIR, CACHE_DIR


def load_representation_for_downstream_application(
    name, representation, update_cache: bool = False
): 
    
    suffix = ["representations", name, f"{representation}.parquet"]
    
    lpath = dm.fs.join(CACHE_DIR, "downstream_applications", *suffix)
    if not dm.fs.exists(lpath) or update_cache:
        rpath = dm.fs.join(DOWNSTREAM_APPS_DATA_DIR, *suffix)
        dm.fs.copy_file(rpath, lpath)
    
    data = pd.read_parquet(lpath)
    
    X = np.stack(data["representation"].values)
    
    # We filter out feature vectors that are entirely None,
    # but individual features can still be NaN.
    mask = ~np.isnan(X).any(axis=1)
    return X[mask]


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

        
def get_outlier_bounds(X, factor: float = 1.0):
        
    q1 = np.quantile(X, 0.25)
    q3 = np.quantile(X, 0.75)
    iqr = q3 - q1

    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
        
    return lower, upper
