import datamol as dm
from rdkit.Chem.Scaffolds import MurckoScaffold
from loguru import logger


def compute_murcko_scaffold(mol):
    """Computes the Bemis-Murcko scaffold of a compounds."""
    mol = dm.to_mol(mol)
    scaffold = dm.to_scaffold_murcko(mol)
    scaffold = dm.to_smiles(scaffold)
    return scaffold


def compute_generic_scaffold(smi: str):
    """Computes the scaffold (i.e. the domain) for the datapoint. The generic scaffold is the
    structural graph of the Murcko scaffold

    Args:
        smi (str): The SMILES string of the molecule to find the generic scaffold for
    Returns:
        The SMILES of the Generic scaffold of the input SMILES
    """

    scaffold = compute_murcko_scaffold(smi)
    with dm.without_rdkit_log(mute_errors=False):
        try:
            scaffold = dm.to_mol(scaffold)
            scaffold = MurckoScaffold.MakeScaffoldGeneric(mol=scaffold)
            scaffold = dm.to_smiles(scaffold)
        except Exception as exception:
            logger.debug(f"Failed to compute the GenericScaffold for {smi} due to {exception}")
            logger.debug(f"Returning the empty SMILES as the scaffold")
            scaffold = ""
    return scaffold
