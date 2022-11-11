import os
import click 

import pandas as pd
import datamol as dm

from mood.preprocessing import standardize_smiles
from mood.representations import MOOD_REPRESENTATIONS, featurize
from mood.constants import DOWNSTREAM_APPS_DATA_DIR


@click.command()
@click.option("--molecule-set", type=click.Choice(["optimization", "virtual_screening"]), required=True)
@click.option("--representation", type=click.Choice(MOOD_REPRESENTATIONS), required=True)
@click.option("--n-jobs", type=int)
@click.option("--verbose/--silent", default=False)
@click.option("--override/--no-override", default=False)
def cli(molecule_set, representation, n_jobs, verbose, override): 
    
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


if __name__ == "__main__": 
    cli()
