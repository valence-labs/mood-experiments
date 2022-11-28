import os
import typer

import pandas as pd
import datamol as dm
from molfeat.trans.base import MoleculeTransformer

from loguru import logger
from functools import partial
from typing import Optional
from mood.preprocessing import DEFAULT_PREPROCESSING
from mood.representations import MOOD_REPRESENTATIONS, featurize, TEXTUAL_FEATURIZERS
from mood.constants import DOWNSTREAM_APPS_DATA_DIR, DATASET_DATA_DIR


STATE_DICT = {
    "_molfeat_version": "0.5.2",
    "args": {
        "max_length": 256,
        "name": "pcqm4mv1_graphormer_base",
        "pooling": "mean",
        "precompute_cache": False,
        "version": None,
    },
    "name": "GraphormerTransformer",
},


def cli(
    batch_size: int = 16,
    verbose: bool = False, 
    override: bool = False
): 
    
    graphormer = MoleculeTransformer.from_state_dict(STATE_DICT)
    standardize_fn = DEFAULT_PREPROCESSING["Graphormer"]

    # Precompute Graphormer for the downstream applications
    for molecule_set in SUPPORTED_DOWNSTREAM_APPS:
        
        in_path = dm.fs.join(DOWNSTREAM_APPS_DATA_DIR, f"{molecule_set}.csv")
        out_path = dm.fs.join(
            DOWNSTREAM_APPS_DATA_DIR, 
            "representations", 
            molecule_set, 
            f"{representation}.parquet"
        )

        if dm.fs.exists(out_path) and not override: 
            raise ValueError(f"{out_path} already exists! Use --override to override!")
    
        # Load
        logger.info(f"Loading SMILES from {in_path}")
        df = pd.read_csv(in_path)

        # Standardization
        df["smiles"] = dm.utils.parallelized(
            standardize_fn,
            df["canonical_smiles"].values,
            n_jobs=-1,
            progress=verbose,
        )
    
        # Compute the representation
        logger.info(f"Precomputing {representation} representation")
        feats = graphormer.batch_transform(graphormer, df["smiles"].values, batch_size=batch_size, n_jobs=None)
        
        df["representation"] = list(feats)
        df = df[~pd.isna(df["representation"])]
    
        # Save
        logger.info(f"Saving results to {out_path}")
        df[["unique_id", "representation"]].to_parquet(out_path)
        
        
    for dataset, (smiles, _) in dataset_iterator(progress=verbose, whitelist=dataset, disable_logs=True):
        
        out_path = dm.fs.join(
            DATASET_DATA_DIR,
            "representations", 
            dataset, 
            f"{representation}.parquet"
        )
        
        if dm.fs.exists(out_path) and not override: 
            raise ValueError(f"{out_path} already exists! Use --override to override!")
        
        df = pd.DataFrame()
        df["smiles"] = dm.utils.parallelized(
            standardize_fn,
            smiles,
            n_jobs=-1,
            progress=verbose,
        )
        
        df["unique_id"] = dm.utils.parallelized(
            dm.unique_id,
            df["smiles"].values,
            n_jobs=-1,
            progress=verbose,
        )
        
        feats = graphormer.batch_transform(graphormer, df["smiles"].values, batch_size=batch_size, n_jobs=None)
        df["representation"] = list(feats)
        
        # Save
        logger.info(f"Saving results to {out_path}")
        df.to_parquet(out_path)
        
        
if __name__ == "__main__":
    typer.run(cli)