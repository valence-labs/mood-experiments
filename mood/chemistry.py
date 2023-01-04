import datamol as dm


def compute_murcko_scaffold(mol):
    """Computes the Bemis-Murcko scaffold of a compounds."""
    mol = dm.to_mol(mol)
    scaffold = dm.to_scaffold_murcko(mol)
    scaffold = dm.to_smiles(scaffold)
    return scaffold
