import datamol as dm
import numpy as np

from collections import OrderedDict
from typing import Optional, List
from functools import partial
from copy import deepcopy

from rdkit import Chem
from rdkit.Chem.QED import properties
from rdkit.Chem import Descriptors
from rdkit.Chem import Descriptors3D
from rdkit.Chem import rdchem
from rdkit.Chem import FindMolChiralCenters
from rdkit.Chem import rdPartialCharges
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem



def representation_iterator(
    smiles, 
    n_jobs: Optional[int] = None, 
    progress: bool = True,
    mask_nan: bool = True,
    return_mask: bool = True,
    disable_logs: bool = True,
    whitelist: Optional[List[str]] = None,
    blacklist: Optional[List[str]] = None,
):
    
    if whitelist is not None and blacklist is not None: 
        msg = "You cannot use a blacklist and whitelist at the same time"
        raise ValueError(msg)
    
    all_representations = MOOD_REPRESENTATIONS
    
    if whitelist is not None: 
        all_representations = [d for d in all_representations if d in whitelist]
    if blacklist is not None: 
        all_representations = [d for d in all_representations if d not in blacklist]
    
    for name in all_representations:
        
        feats = featurize(smiles, name, n_jobs, progress, disable_logs)
        
        # Mask out invalid features
        mask = [
            i for i, a in enumerate(feats) 
            if a is not None 
            and ~np.isnan(a).any()
        ]
        
        if mask_nan: 
            feats = feats[mask]
            
        # If the array had any Nones, it would not be a proper
        # 2D array so we convert to one here.
        feats = np.stack(feats)
        
        if return_mask:
            feats = feats, mask
        yield name, feats


def featurize(
    smiles, 
    name,
    n_jobs: Optional[int] = None, 
    progress: bool = True,
    disable_logs: bool = False
):
    if name not in _REPR_TO_FUNC:
        msg = f"{name} is not supported. Choose from {MOOD_REPRESENTATIONS}"
        raise NotImplementedError(name)
    
    fn = _REPR_TO_FUNC[name]
    fn = partial(fn, disable_logs=disable_logs)
    
    reprs = dm.utils.parallelized(
        fn, smiles, n_jobs=n_jobs, progress=progress, tqdm_kwargs={"desc": name}
    )
    
    return np.array(reprs)


def compute_whim(smi, disable_logs: bool = False):
    """
    Compute a WHIM descriptor from a RDkit molecule object
    Code adapted from MoleculeACE, Van Tilborg et al. (2022)
    """
    with dm.without_rdkit_log(enable=disable_logs):
        mol = dm.to_mol(smi)
        if mol is None:
            # Failed
            return

        mol = Chem.AddHs(mol)

        # Use distance geometry to obtain initial coordinates for a molecule
        ret = AllChem.EmbedMolecule(
            mol, useRandomCoords=True, useBasicKnowledge=True, randomSeed=0, clearConfs=True, maxAttempts=5
        )
        if ret == -1:
            # Failed
            return

        AllChem.MMFFOptimizeMolecule(mol, maxIters=1000, mmffVariant="MMFF94")

        # calculate WHIM 3D descriptor
        whim = rdMolDescriptors.CalcWHIM(mol)
        whim = np.array(whim).astype(np.float32)
    return whim


def _compute_extra_2d_features(mol):
    """Computes some additional descriptors besides the default ones RDKit offers""" 
    mol = deepcopy(mol)
    FindMolChiralCenters(mol, force=True)
    p_obj = rdMolDescriptors.Properties()
    props = OrderedDict(zip(p_obj.GetPropertyNames(), p_obj.ComputeProperties(mol)))
    qed_props = properties(mol)
    props["Alerts"] = qed_props.ALERTS
    return props


def _charge_descriptors_fix(mol: dm.Mol):
    """Recompute the RDKIT 2D Descriptors related to charge
    
    We change the procedure: 
    1. We disconnect the metal from the molecule
    2. We add the hydrogen atoms 
    3. We make sure that gasteiger is recomputed. 
    
    This fixes an issue where these descriptors could be NaN or Inf,
    while also making sure we are closer to the proper interpretation
    """
    descrs = {}
    mol = dm.standardize_mol(mol, disconnect_metals=True)
    mol = dm.add_hs(mol, explicit_only=False)
    rdPartialCharges.ComputeGasteigerCharges(mol)
    atomic_charges = [float(at.GetProp("_GasteigerCharge")) for at in mol.GetAtoms()]
    atomic_charges = np.clip(atomic_charges, a_min=-500, a_max=500)
    min_charge, max_charge = np.nanmin(atomic_charges), np.nanmax(atomic_charges)
    descrs["MaxPartialCharge"] = max_charge
    descrs["MinPartialCharge"] = min_charge
    descrs["MaxAbsPartialCharge"] = max(np.abs(min_charge), np.abs(max_charge))
    descrs["MinAbsPartialCharge"] = min(np.abs(min_charge), np.abs(max_charge))
    return descrs


def compute_desc2d(smi, disable_logs: bool = False):
    
    descr_fns = {name: fn for (name, fn) in Descriptors.descList}
    
    all_features = [d[0] for d in Descriptors.descList]
    all_features += [
        "NumAtomStereoCenters",
        "NumUnspecifiedAtomStereoCenters",
        "NumBridgeheadAtoms",
        "NumAmideBonds",
        "NumSpiroAtoms",
        "Alerts",
    ]
    
    with dm.without_rdkit_log(enable=disable_logs):
        mol = dm.to_mol(smi)
        descr_extra = _compute_extra_2d_features(mol)
        descr_charge = _charge_descriptors_fix(mol)
    
        descr = []

        for name in all_features:
            val = float("nan")
            if name in descr_charge:
                val = descr_charge[name]
            elif name == "Ipc":
                # Fixes a bug for the RDKit IPC value. For context, see:
                # https://github.com/rdkit/rdkit/issues/1527
                val = descr_fns[name](mol, avg=True)
            elif name in descr_fns:
                val = descr_fns[name](mol)
            else:
                assert name in descr_extra
                val = descr_extra[name]
            descr.append(val)

        descr = np.asarray(descr)
    return descr


def compute_ecfp6(smi, disable_logs: bool = False):
    with dm.without_rdkit_log(enable=disable_logs):
        return dm.to_fp(smi, fp_type="ecfp", radius=3)


def compute_maccs(smi, disable_logs: bool = False):
    with dm.without_rdkit_log(enable=disable_logs):
        return dm.to_fp(smi, fp_type="maccs")


# TODO: When adding Graphormer and ChemBERTa,
#     ensure we use the appropiate preprocessing fn
_REPR_TO_FUNC = {   
    "MACCS": compute_maccs, 
    "ECFP6": compute_ecfp6,
    "Desc2D": compute_desc2d,
    "WHIM": compute_whim,
}    

MOOD_REPRESENTATIONS = list(_REPR_TO_FUNC.keys())