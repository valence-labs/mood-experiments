import datamol as dm
from functools import partial
from rdkit.Chem import SaltRemover


def standardize_smiles(smi, for_text_based_model: bool = False, disable_logs: bool = False):
    """A good default standardization function for fingerprints and GNNs"""
    
    with dm.without_rdkit_log(enable=disable_logs):
        mol = dm.to_mol(smi, ordered=True, sanitize=False)
        mol = dm.sanitize_mol(mol)
        
        if for_text_based_model:
            mol = dm.standardize_mol(mol)
            
        else: 
            mol = dm.standardize_mol(mol, disconnect_metals=True)
            remover = SaltRemover.SaltRemover()
            mol = remover.StripMol(mol, dontRemoveEverything=True)
            
    return dm.to_smiles(mol)


DEFAULT_PREPROCESSING = {
    "MACCS": partial(standardize_smiles, for_text_based_model=False), 
    "ECFP6": partial(standardize_smiles, for_text_based_model=False),
    "Desc2D": partial(standardize_smiles, for_text_based_model=False),
    "WHIM": partial(standardize_smiles, for_text_based_model=False),
    "ChemBERTa": partial(standardize_smiles, for_text_based_model=True),
}
