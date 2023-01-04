from copy import deepcopy

import datamol as dm
import numpy as np

from typing import Optional, List

import torch.utils.data
from tdc.single_pred import ADME, Tox
from tdc.metadata import dataset_names
from torch.utils.data import default_collate

from mood.chemistry import compute_generic_scaffold
from mood.constants import CACHE_DIR


class SimpleMolecularDataset(torch.utils.data.Dataset):
    """
    Simple wrapper to use these datasets with PyTorch (Lightning)
    Special is that this includes the notion of a molecular domain,
    needed for Domain Generalization
    """

    def __init__(self, smiles, X, y):
        self.smiles = smiles
        self.X = X
        self.y = y
        self.random_state = None
        self.domains = np.array([compute_generic_scaffold(smi) for smi in self.smiles])

    def __getitem__(self, index):
        x = self.X[index]
        if self.domains is not None:
            x = (x, self.domains[index])
        return x, self.y[index]

    def __len__(self):
        return len(self.X)

    def filter_by_indices(self, indices):
        cpy = deepcopy(self)
        cpy.smiles = cpy.smiles[indices]
        cpy.X = cpy.X[indices]
        cpy.y = cpy.y[indices]
        if cpy.domains is not None:
            cpy.domains = cpy.domains[indices]
        return cpy


class DAMolecularDataset(torch.utils.data.Dataset):
    """Simple wrapper that creates a dataset with a supervised source domain and unsupervised target domain.
    This is needed for Domain Adaptation algorithms."""

    def __init__(self, source_dataset: SimpleMolecularDataset, target_dataset: SimpleMolecularDataset):
        self.src = source_dataset
        self.tgt = target_dataset

    def __getitem__(self, item):
        src = self.src.__getitem__(item)
        tgt_index = np.random.default_rng(item).integers(0, len(self.tgt))
        (x, domain), y = self.tgt.__getitem__(tgt_index)
        return {"source": src, "target": (x, domain)}

    def __len__(self):
        return len(self.src)


def domain_based_collate(batch):
    """Custom collate function that splits a single batch into several mini-batches based on the domain.
    Doing that here instead of on the GPU is faster."""
    domains = [domain for (X, domain), y in batch]
    _, inverse = np.unique(domains, return_inverse=True, axis=0)

    mini_batches = []
    for idx in np.unique(inverse):
        indices = np.flatnonzero(inverse == idx)
        mini_batch = default_collate([batch[i] for i in indices])
        mini_batches.append(mini_batch)
    return mini_batches


def load_data_from_tdc(name: str, disable_logs: bool = False):
    """
    Endpoint from loading a dataset from TDC
    """

    original_name = name
    if name in MOOD_TO_TDC:
        name = MOOD_TO_TDC[name]

    path = dm.fs.join(CACHE_DIR, "TDC")

    # Load the dataset
    if name.lower() in dataset_names["ADME"]:
        dataset = ADME(name=name, path=path)
    elif name.lower() in dataset_names["Tox"]:
        dataset = Tox(name=name, path=path)
    else:
        msg = f"{original_name} is not supported. Choose from {MOOD_DATASETS}."
        raise RuntimeError(msg)

    # Standardize the SMILES
    with dm.without_rdkit_log(enable=disable_logs):
        smiles = dataset.entity1
        smiles = np.array([dm.to_smiles(dm.to_mol(smi)) for smi in smiles])

    # Load the targets
    y = np.array(dataset.y)

    # Mask out NaN that might be the result of standardization
    mask = [i for i, x in enumerate(smiles) if x is not None]
    smiles = smiles[mask]
    y = y[mask]

    return smiles, y


def dataset_iterator(
    disable_logs: bool = True,
    whitelist: Optional[List[str]] = None,
    blacklist: Optional[List[str]] = None,
):
    """Endpoint for iterating over several datasets from TDC"""

    if whitelist is not None and blacklist is not None:
        msg = "You cannot use a blacklist and whitelist at the same time"
        raise ValueError(msg)

    all_datasets = MOOD_DATASETS

    if whitelist is not None:
        all_datasets = [d for d in all_datasets if d in whitelist]
    if blacklist is not None:
        all_datasets = [d for d in all_datasets if d not in blacklist]

    for name in all_datasets:
        yield name, load_data_from_tdc(name, disable_logs)


TDC_TO_MOOD = {
    "BBB_Martins": "BBB",
    "CYP2C9_Veith": "CYP2C9",
    "Caco2_Wang": "Caco-2",
    "Clearance_Hepatocyte_AZ": "Clearance",
    "DILI": "DILI",
    "HIA_Hou": "HIA",
    "Half_Life_Obach": "HalfLife",
    "Lipophilicity_AstraZeneca": "Lipophilicity",
    "PPBR_AZ": "PPBR",
    "Pgp_Broccatelli": "Pgp",
    "hERG": "hERG",
}

# Ordered by size
MOOD_DATASETS = [
    "DILI",
    "HIA",
    "hERG",
    "HalfLife",
    "Caco-2",
    "Clearance",
    "Pgp",
    "PPBR",
    "BBB",
    "Lipophilicity",
    "CYP2C9",
]

MOOD_TO_TDC = {v: k for k, v in TDC_TO_MOOD.items()}
MOOD_CLSF_DATASETS = ["BBB", "CYP2C9", "DILI", "HIA", "Pgp", "hERG"]
MOOD_REGR_DATASETS = [d for d in MOOD_DATASETS if d not in MOOD_CLSF_DATASETS]
