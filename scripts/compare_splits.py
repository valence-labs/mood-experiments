import fsspec
import numpy as np
import pandas as pd
import datamol as dm

from loguru import logger
from typing import Optional, List
from datetime import datetime

from mood.dataset import dataset_iterator
from mood.representations import representation_iterator
from mood.splitter import MOODSplitter, get_mood_splitters
from mood.preprocessing import DEFAULT_PREPROCESSING
from mood.utils import load_distances_for_downstream_application, save_figure_with_fsspec
from mood.constants import DOWNSTREAM_RESULTS_DIR
from mood.distance import get_metric


def cli(
    base_save_dir: str = DOWNSTREAM_RESULTS_DIR,
    sub_save_dir: Optional[str] = None,
    skip_representation: Optional[List[str]] = None,
    skip_dataset: Optional[List[str]] = None,
    save_figures: bool = True,
    n_splits: int = 5,
    seed: Optional[int] = 0,
    use_cache: bool = False,
    batch_size: int = 16,
    verbose: bool = False,
    overwrite: bool = False,
    n_jobs: Optional[int] = None,
):

    df = pd.DataFrame()

    if sub_save_dir is None:
        sub_save_dir = datetime.now().strftime("%Y%m%d")

    fig_out_dir = dm.fs.join(base_save_dir, "figures", "compare_splits", sub_save_dir)
    dm.fs.mkdir(fig_out_dir, exist_ok=True)
    
    data_out_dir = dm.fs.join(base_save_dir, "numpy", "compare_splits", sub_save_dir)
    dm.fs.mkdir(data_out_dir, exist_ok=True)

    dataset_it = dataset_iterator(blacklist=skip_dataset)
    for dataset, (smiles, y) in dataset_it:

        representation_it = representation_iterator(
            smiles,
            n_jobs=n_jobs,
            progress=verbose,
            standardize_fn=DEFAULT_PREPROCESSING,
            batch_size=batch_size,
            blacklist=skip_representation,
        )

        for representation, (X, mask) in representation_it:

            logger.info(f"Loading precomputed distances for virtual screening")
            distances_vs = load_distances_for_downstream_application(
                "virtual_screening", representation, dataset, update_cache=not use_cache
            )

            logger.info(f"Loading precomputed distances for optimization")
            distances_op = load_distances_for_downstream_application(
                "optimization", representation, dataset, update_cache=not use_cache
            )

            metric = get_metric(X)
            if metric == "jaccard":
                X = X.astype(bool)

            splitters = get_mood_splitters(smiles[mask], n_splits, seed, n_jobs=n_jobs)
            splitter = MOODSplitter(splitters, np.concatenate((distances_vs, distances_op)), metric, k=5)
            splitter.fit(X, progress=verbose, plot=save_figures)

            if save_figures:
                out_path = dm.fs.join(fig_out_dir, f"fig_{dataset}_{representation}.png")
                if not overwrite and dm.fs.exists(out_path):
                    raise RuntimeError(
                        f"{out_path} already exists. Specify a different path or use --overwrite"
                    )
                logger.info(f"Saving figure to {out_path}")
                save_figure_with_fsspec(out_path, exist_ok=overwrite)
            
            for char in splitter._split_chars:
                out_path = dm.fs.join(data_out_dir, f"distances_{dataset}_{representation}_{char.label}.npy")
                if not overwrite and dm.fs.exists(out_path):
                    raise RuntimeError(
                        f"{out_path} already exists. Specify a different path or use --overwrite"
                    )
                logger.info(f"Saving distance data to {out_path}")
                with fsspec.open(out_path, "wb") as fd:
                    np.save(fd, char.distances)
            
            df_ = splitter.get_protocol_results()
            df_["representation"] = representation
            df_["dataset"] = dataset

            df = pd.concat((df, df_), ignore_index=True)

    out_dir = dm.fs.join(base_save_dir, "dataframes", "compare_splits", sub_save_dir)
    dm.fs.mkdir(out_dir, exist_ok=True)

    out_path = dm.fs.join(out_dir, "splits.csv")
    if not overwrite and dm.fs.exists(out_path):
        raise RuntimeError(f"{out_path} already exists. Specify a different path or use --overwrite")

    df.to_csv(out_path, index=False)
    logger.info(f"Saving dataframe to {out_path}")
