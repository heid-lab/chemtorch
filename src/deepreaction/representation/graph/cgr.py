from typing import List, Optional, Tuple

import hydra
import torch

from torch_geometric.data import Data
from omegaconf import DictConfig
from rdkit import Chem

from deepreaction.representation.graph.graph_reprs_utils import (
    AtomOriginType,
    EdgeOriginType,
    make_mol,
    map_reac_to_prod,
)
from deepreaction.representation.representation_base import RepresentationBase


class CGR(RepresentationBase[Data]):
    """
    Stateless class for constructing Condensed Graph of Reaction (CGR) representations.

    This class does not hold any data itself. Instead, it provides a `forward()` method
    that takes a sample (e.g., a dict or pd.Series) and returns a PyTorch Geometric Data object
    representing the reaction as a graph.

    # TODO: Update docstring once featurizers are passed explicitly
    Usage:
        cgr = CGR(featurizer_cfg)
        data = cgr.forward(sample)
    """

    def __init__(
        self,
        atom_featurizer,
        bond_featurizer,
        *args,
        **kwargs,
    ):
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer


    # override
    def construct(self, smiles: str, label: Optional[float] = None) -> Data:
        """
        Construct a CGR graph from the sample.
        """
        smiles_reac, _, smiles_prod = smiles.split(">")

        mol_reac, _ = make_mol(smiles_reac)
        mol_prod, _ = make_mol(smiles_prod)

        ri2pi = map_reac_to_prod(mol_reac, mol_prod)

        n_atoms = mol_reac.GetNumAtoms()

        f_atoms_list: List[List[float]] = []
        atom_origin_type_list: List[AtomOriginType] = []

        for i in range(n_atoms):
            f_atoms_list.append(self._get_atom_features_internal(mol_reac, mol_prod, ri2pi, i))
            atom_origin_type_list.append(AtomOriginType.REACTANT_PRODUCT)

        edge_index_list: List[Tuple[int, int]] = []
        f_bonds_list: List[List[float]] = []
        edge_origin_type_list: List[EdgeOriginType] = []

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):  # Iterate over unique pairs
                bond_reac = mol_reac.GetBondBetweenAtoms(i, j)

                # Get corresponding product atom indices
                # These must exist due to checks in map_reac_to_prod
                idx_prod_i = ri2pi[i]
                idx_prod_j = ri2pi[j]
                bond_prod = mol_prod.GetBondBetweenAtoms(idx_prod_i, idx_prod_j)

                if bond_reac is None and bond_prod is None:  # No bond in either
                    continue

                f_bond = self._get_bond_features_internal(bond_reac, bond_prod)

                # Add edges in both directions for an undirected graph
                edge_index_list.append((i, j))
                f_bonds_list.append(f_bond)
                edge_origin_type_list.append(EdgeOriginType.REACTANT_PRODUCT)

                edge_index_list.append((j, i))
                f_bonds_list.append(f_bond)
                edge_origin_type_list.append(EdgeOriginType.REACTANT_PRODUCT)

        # --- Assign to PyG Data attributes ---
        data = Data()
        data.x = torch.tensor(f_atoms_list, dtype=torch.float)

        if edge_index_list:
            data.edge_index = (
                torch.tensor(edge_index_list, dtype=torch.long)
                .t()
                .contiguous()
            )
            data.edge_attr = torch.tensor(f_bonds_list, dtype=torch.float)
            data.edge_origin_type = torch.tensor(
                edge_origin_type_list, dtype=torch.long
            )
        else:  # Handle graphs with no edges
            data.edge_index = torch.empty((2, 0), dtype=torch.long)
            # Determine bond feature dimension for empty edge_attr
            dummy_bond_feat_len = len(
                self._get_bond_features_internal(None, None)
            )
            data.edge_attr = torch.empty(
                (0, dummy_bond_feat_len), dtype=torch.float
            )
            data.edge_origin_type = torch.empty((0), dtype=torch.long)

        if label is not None:
            data.y = torch.tensor([label], dtype=torch.float)
        data.smiles = smiles  # Store original reaction SMILES
        data.atom_origin_type = torch.tensor(
            atom_origin_type_list, dtype=torch.long
        )

        # num_nodes is a standard PyG attribute
        data.num_nodes = n_atoms

        return data
        

    def _get_atom_features_internal(
        self, mol_reac: Chem.Mol, mol_prod: Chem.Mol, ri2pi: List[int], atom_idx: int
    ) -> List[float]:
        """Helper to generate CGR atom features using temporary internal attributes."""
        f_atom_reac = self.atom_featurizer(mol_reac.GetAtomWithIdx(atom_idx))
        # ri2pi[atom_idx] gives the corresponding product atom index
        f_atom_prod = self.atom_featurizer(mol_prod.GetAtomWithIdx(ri2pi[atom_idx]))
        f_atom_diff = [y - x for x, y in zip(f_atom_reac, f_atom_prod)]
        return f_atom_reac + f_atom_diff


    def _get_bond_features_internal(
        self, bond_reac: Optional[Chem.Bond], bond_prod: Optional[Chem.Bond]
    ) -> List[float]:
        """Helper to generate CGR bond features using temporary internal attributes."""
        # Assuming self.bond_featurizer(None) returns a list of default values (e.g., zeros)
        # of the correct base feature length. This is used to determine padding length.
        base_bond_feat_len = len(self.bond_featurizer(None))

        f_bond_reac = (
            self.bond_featurizer(bond_reac)
            if bond_reac
            else [0.0]
            * base_bond_feat_len  # Use float for consistency with PyTorch tensors
        )
        f_bond_prod = (
            self.bond_featurizer(bond_prod)
            if bond_prod
            else [0.0] * base_bond_feat_len
        )
        f_bond_diff = [y - x for x, y in zip(f_bond_reac, f_bond_prod)]
        return f_bond_reac + f_bond_diff
