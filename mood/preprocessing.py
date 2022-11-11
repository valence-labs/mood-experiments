import datamol as dm
from rdkit.Chem import SaltRemover


def standardize_smiles(smi, for_text_based_model: bool = False):
    """A good default standardization function"""
    
    with dm.without_rdkit_log():
        mol = dm.to_mol(smi, ordered=True, sanitize=False)
        mol = dm.sanitize_mol(mol)
        
        if for_text_based_model:
            mol = dm.standardize_mol(mol)
            
        else: 
            mol = dm.standardize_mol(mol, disconnect_metals=True)
            remover = SaltRemover.SaltRemover()
            mol = remover.StripMol(mol, dontRemoveEverything=True)
            
    return dm.to_smiles(mol)
