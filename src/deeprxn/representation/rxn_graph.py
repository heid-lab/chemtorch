from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Dict, List, Tuple

from omegaconf import DictConfig
from rdkit import Chem


class AtomOriginType(IntEnum):
    REACTANT = 0
    PRODUCT = 1
    DUMMY = 2
    REACTANT_PRODUCT = 3


class RxnGraphBase(ABC):
    """Base class for reaction graphs."""

    def __init__(
        self,
        smiles: str,
        atom_featurizer: callable,
        bond_featurizer: callable,
    ):
        """Initialize reaction graph.

        Args:
            smiles: reaction string with atom mapping
            atom_featurizer: Function to generate atom features
            bond_featurizer: Function to generate bond features
        """
        self.smiles = smiles
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer

    @staticmethod
    def _make_mol(smi: str) -> Tuple[Chem.Mol, List[int]]:
        """Create RDKit mol with atom mapping."""
        params = Chem.SmilesParserParams()
        params.removeHs = False

        parts = smi.split(".")
        atom_origins = []
        current_atom_idx = 0

        for i, part in enumerate(parts):
            mol = Chem.MolFromSmiles(part, params)
            if mol is None:
                continue
            atom_origins.extend([i] * mol.GetNumAtoms())
            current_atom_idx += mol.GetNumAtoms()

        return Chem.MolFromSmiles(smi, params), atom_origins

    def _map_reac_to_prod(self) -> Dict[int, int]:
        """Map reactant atom indices to product atom indices."""
        prod_map_to_id = {
            atom.GetAtomMapNum(): atom.GetIdx()
            for atom in self.mol_prod.GetAtoms()
        }
        return {
            atom.GetIdx(): prod_map_to_id[atom.GetAtomMapNum()]
            for atom in self.mol_reac.GetAtoms()
        }

    @abstractmethod
    def _build_graph(self):
        """Build the reaction graph representation.

        To be implemented by specific graph types (CGR, Connected Pair).
        Should populate:
            - self.f_atoms
            - self.f_bonds
            - self.edge_index
        """
        pass
