import os
import typer

import pandas as pd
import datamol as dm

from typing import Optional, List
from mood.preprocessing import standardize_smiles
from mood.dataset import dataset_iterator
from mood.representations import representation_iterator
from mood.constants import DOWNSTREAM_APPS_DATA_DIR
from mood.utils import load_representation_for_downstream_application
from mood.distance import compute_knn_distance


def save(distances, compounds, molecule_set, dataset, representation, overwrite):
    
    out_path = dm.fs.join(
        DOWNSTREAM_APPS_DATA_DIR, 
        "distances", 
        molecule_set,
        dataset,
        f"{representation}.parquet"
    )
    if dm.fs.exists(out_path) and not overwrite:
        raise RuntimeError(f"{out_path} already exists!")
    
    df = pd.DataFrame({"unique_id": compounds, "distance": distances})
    df.to_parquet(out_path)
    return df


def cli(
    overwrite: bool = False,
    representation: Optional[List[str]] = None,
    dataset: Optional[List[str]] = None,
): 
    
    if len(dataset) == 0:
        dataset = None
    if len(representation) == 0:
        representation = None
        
    for dataset, (smiles, y) in dataset_iterator(standardize_smiles, progress=True, whitelist=dataset):
        
        for representation, (X, mask) in representation_iterator(smiles, n_jobs=-1, progress=True, whitelist=representation):

            y_repr = y[mask]

            virtual_screening, vs_compounds = load_representation_for_downstream_application(
                "virtual_screening", representation, return_compound_ids=True
            )
            optimization, opt_compounds = load_representation_for_downstream_application(
                "optimization", representation, return_compound_ids=True
            )

            input_distances = compute_knn_distance(X, [optimization, virtual_screening], n_jobs=-1)
            
            save(input_distances[0], opt_compounds, "optimization", dataset, representation, overwrite)
            save(input_distances[1], vs_compounds, "virtual_screening", dataset, representation, overwrite)
