import datamol as dm
import numpy as np

from typing import Optional, Callable
from tdc.single_pred import ADME, Tox
from tdc.metadata import dataset_names

from mood.constants import CACHE_DIR


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
MOOD_TO_TDC = {v: k for k, v in TDC_TO_MOOD.items()}
MOOD_DATASETS = list(MOOD_TO_TDC.keys())


def load_data_from_tdc(name: str, standardize_fn: Optional[Callable] = None):
    
    if name in MOOD_TO_TDC:
        name = MOOD_TO_TDC[name]
    
    path = dm.fs.join(CACHE_DIR, "TDC")
    
    # Load the dataset
    if name.lower() in dataset_names["ADME"]:
        dataset = ADME(name=name, path=path)
    elif name.lower() in dataset_names["Tox"]:
        dataset = Tox(name=name, path=path)
    else: 
        msg = f"{name} is not supported. Choose from {MOOD_DATASETS}."
        raise RuntimeError(msg)
    
    # Standardize the SMILES
    smiles = dataset.entity1.to_numpy()
    smiles = np.array([dm.to_smiles(dm.to_mol(smi)) for smi in smiles])
    if standardize_fn is not None:
        smiles = np.array(dm.utils.parallelized(standardize_fn, smiles))

    # Load the targets
    y = dataset.y.to_numpy()
    
    # Mask out NaN that might be the result of standardization
    mask = [i for i, x in enumerate(smiles) if x is not None]
    smiles = smiles[mask]
    y = y[mask]
    
    return smiles, y


def dataset_iterator(standardize_fn: Optional[Callable] = None):
    for name in MOOD_DATASETS:
        yield name, load_data_from_tdc(name, standardize_fn)
