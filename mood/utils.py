import datamol as dm
import pandas as pd
import numpy as np

from mood.constants import DOWNSTREAM_APPS_DATA_DIR


def load_representation_for_downstream_application(name, representation): 
    
    path = dm.fs.join(
        DOWNSTREAM_APPS_DATA_DIR, 
        "representations",
        name, 
        f"{representation}.parquet"
    )
    
    data = pd.read_parquet(path)
    X = np.stack(data["representation"].values)
    
    # We filter out feature vectors that are entirely None,
    # but individual features can still be NaN.
    mask = ~np.isnan(X).any(axis=1)
    return X[mask]
