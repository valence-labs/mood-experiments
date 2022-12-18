import typer

import pandas as pd
import datamol as dm

from loguru import logger
from typing import Optional
from mood.preprocessing import DEFAULT_PREPROCESSING
from mood.representations import MOOD_REPRESENTATIONS, featurize
from mood.constants import DOWNSTREAM_APPS_DATA_DIR, SUPPORTED_DOWNSTREAM_APPS


def cli(
    molecule_set: str,
    representation: str,
    n_jobs: Optional[int] = None,
    batch_size: int = 16,
    verbose: bool = False,
    override: bool = False,
):

    if molecule_set not in SUPPORTED_DOWNSTREAM_APPS:
        raise typer.BadParameter(f"--molecule-set should be one of {SUPPORTED_DOWNSTREAM_APPS}.")
    if representation not in MOOD_REPRESENTATIONS:
        raise typer.BadParameter(f"--representation should be one of {MOOD_REPRESENTATIONS}.")

    in_path = dm.fs.join(DOWNSTREAM_APPS_DATA_DIR, f"{molecule_set}.csv")
    out_path = dm.fs.join(
        DOWNSTREAM_APPS_DATA_DIR, "representations", molecule_set, f"{representation}.parquet"
    )

    if dm.fs.exists(out_path) and not override:
        raise ValueError(f"{out_path} already exists! Use --override to override!")

    # Load
    logger.info(f"Loading SMILES from {in_path}")
    df = pd.read_csv(in_path)

    # Standardization fn
    standardize_fn = DEFAULT_PREPROCESSING[representation]

    # Compute the representation
    logger.info(f"Precomputing {representation} representation")
    df["representation"] = list(
        featurize(
            df["canonical_smiles"].values,
            representation,
            standardize_fn,
            n_jobs=n_jobs,
            progress=verbose,
            batch_size=batch_size,
            return_mask=False,
            disable_logs=True,
        )
    )
    df = df[~pd.isna(df["representation"])]

    # Save
    logger.info(f"Saving results to {out_path}")
    df[["unique_id", "representation"]].to_parquet(out_path)
