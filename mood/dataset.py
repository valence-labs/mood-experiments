import datamol as dm
import numpy as np

from functools import partial
from typing import Optional, Callable, List
from tdc.single_pred import ADME, Tox
from tdc.metadata import dataset_names
from mood.constants import CACHE_DIR


def load_data_from_tdc(
    name: str, 
    progress: bool = True,
    disable_logs: bool = False
):
    
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
    progress: bool = False,
    whitelist: Optional[List[str]] = None,
    blacklist: Optional[List[str]] = None,
):
    
    if whitelist is not None and blacklist is not None: 
        msg = "You cannot use a blacklist and whitelist at the same time"
        raise ValueError(msg)
    
    all_datasets = MOOD_DATASETS
    
    if whitelist is not None: 
        all_datasets = [d for d in all_datasets if d in whitelist]
    if blacklist is not None: 
        all_datasets = [d for d in all_datasets if d not in blacklist]
    
    for name in all_datasets:
        yield name, load_data_from_tdc(name, progress, disable_logs)


TDC_TO_MOOD = {
    "BBB_Martins": "BBB",
    "CYP2C9_Veith": "CYPP4502C9",
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
   'DILI',
   'HIA',
   'hERG',
   'HalfLife',
   'Caco-2',
   'Clearance',
   'Pgp',
   'PPBR',
   'BBB',
   'Lipophilicity',
   'CYPP4502C9'
]

MOOD_TO_TDC = {v: k for k, v in TDC_TO_MOOD.items()}
MOOD_CLSF_DATASETS = ["BBB", "CYPP4502C9", "DILI", "HIA", "Pgp", "hERG"]
MOOD_REGR_DATASETS = [d for d in MOOD_DATASETS if d not in MOOD_CLSF_DATASETS]
