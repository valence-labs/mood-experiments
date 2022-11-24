import os
import typer

import pandas as pd
import datamol as dm

from typing import Optional
from mood.preprocessing import standardize_smiles
from mood.representations import MOOD_REPRESENTATIONS, featurize
from mood.constants import DOWNSTREAM_APPS_DATA_DIR, SUPPORTED_DOWNSTREAM_APPS


def cli(
    molecule_set: str, 
    representation: str, 
    n_jobs: Optional[int] = None, 
    verbose: bool = False, 
    override: bool = False
): 
    
    if molecule_set not in SUPPORTED_DOWNSTREAM_APPS:
        raise typer.BadParameter(f"--molecule-set should be one of {SUPPORTED_DOWNSTREAM_APPS}.")
    if representation not in MOOD_REPRESENTATIONS:
        raise typer.BadParameter(f"--representation should be one of {MOOD_REPRESENTATIONS}.")

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
    df = pd.read_csv(in_path)
    
    # Standardize the SMILES
    df["smiles"] = dm.utils.parallelized(
        standardize_smiles, 
        df["canonical_smiles"].values, 
        n_jobs=n_jobs, 
        progress=verbose
    )
    
    # Compute the representation
    df["representation"] = list(featurize(
        df["smiles"].values, 
        representation, 
        n_jobs=n_jobs, 
        progress=verbose, 
    ))
    df = df[~pd.isna(df["representation"])]
    
    # Save
    df[["unique_id", "representation"]].to_parquet(out_path)
