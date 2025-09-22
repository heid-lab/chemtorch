from enum import IntEnum
import re
from typing import Dict, List, Tuple

from rdkit import Chem


class AtomOriginType(IntEnum):
    REACTANT = 0
    PRODUCT = 1
    DUMMY = 2
    REACTANT_PRODUCT = 3


class EdgeOriginType(IntEnum):
    REACTANT = 0
    PRODUCT = 1
    DUMMY = 2
    REACTANT_PRODUCT = 3


def make_mol(smi: str) -> Tuple[Chem.Mol, List[int]]:
    """Create RDKit mol with atom mapping.

    Args:
        smi: SMILES string

    Returns:
        Tuple containing the RDKit molecule and a list of atom origins
    """
    params = Chem.SmilesParserParams()
    params.removeHs = False

    parts = smi.split(".")
    atom_origins = []
    current_atom_idx = 0

    for i, part in enumerate(parts):
        mol = Chem.MolFromSmiles(part, params)
        if mol is None:  # TODO: look into this
            continue
        atom_origins.extend([i] * mol.GetNumAtoms())
        current_atom_idx += mol.GetNumAtoms()

    return Chem.MolFromSmiles(smi, params), atom_origins


def map_reac_to_prod(mol_reac: Chem.Mol, mol_prod: Chem.Mol) -> Dict[int, int]:
    """Map reactant atom indices to product atom indices.

    Args:
        mol_reac: Reactant molecule
        mol_prod: Product molecule

    Returns:
        Dictionary mapping reactant atom indices to product atom indices
    """
    prod_map_to_id = dict(
        [(atom.GetAtomMapNum(), atom.GetIdx()) for atom in mol_prod.GetAtoms()]
    )
    reac_id_to_prod_id = dict(
        [
            (atom.GetIdx(), prod_map_to_id[atom.GetAtomMapNum()])
            for atom in mol_reac.GetAtoms()
        ]
    )
    return reac_id_to_prod_id

def remove_atom_mapping(smiles: str):
    return re.sub(r':\d+(?=\])', '', smiles)
