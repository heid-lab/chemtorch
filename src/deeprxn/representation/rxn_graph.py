from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import hydra
import torch_geometric as tg
from omegaconf import DictConfig
from rdkit import Chem

from deeprxn.transform.transform import TransformBase


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
        label: float,
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
        self.label = label
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        # self.transforms = self._init_transforms(transform_cfg)

        self.smiles_reac, _, self.smiles_prod = self.smiles.split(">")

        # initialize molecules with atom mapping
        self.mol_reac, self.reac_origins = self._make_mol(self.smiles_reac)
        self.mol_prod, self.prod_origins = self._make_mol(self.smiles_prod)
        self.ri2pi = self._map_reac_to_prod()

        self.f_atoms: List = []  # atom features
        self.f_bonds: List = []  # bond features
        self.edge_index: List[Tuple[int, int]] = []
        self.atom_origin_type: List[AtomOriginType] = []
        self.n_atoms = None

    # def _init_transforms(
    #     self, transform_cfg: Optional[DictConfig]
    # ) -> List[TransformBase]:
    #     """Initialize transformation objects from config."""
    #     if transform_cfg is None:
    #         return []

    #     transforms = []
    #     for _, config in transform_cfg.items():
    #         transform = hydra.utils.instantiate(config)
    #         transforms.append(transform)
    #     return transforms

    # def _apply_transforms(self):
    #     """Apply all registered transformations in order."""
    #     for transform in self.transforms:
    #         transform(self)

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
        prod_map_to_id = dict(
            [
                (atom.GetAtomMapNum(), atom.GetIdx())
                for atom in self.mol_prod.GetAtoms()
            ]
        )
        reac_id_to_prod_id = dict(
            [
                (atom.GetIdx(), prod_map_to_id[atom.GetAtomMapNum()])
                for atom in self.mol_reac.GetAtoms()
            ]
        )
        return reac_id_to_prod_id

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

    @abstractmethod
    def to_pyg_data(self) -> tg.data.Data:
        """Convert the molecular graph to a PyTorch Geometric Data object."""
        pass
