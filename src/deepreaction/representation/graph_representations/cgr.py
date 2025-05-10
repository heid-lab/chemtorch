from typing import List, Optional, Tuple

import hydra
import torch
import torch_geometric as tg
from omegaconf import DictConfig
from rdkit import Chem

from deepreaction.representation.graph_representations.graph_reprs_utils import (
    AtomOriginType,
    EdgeOriginType,
    make_mol,
    map_reac_to_prod,
)


class CGR(tg.data.Data):
    """
    Condensed Graph of Reaction (CGR) representation, a PyTorch Geometric Data.
    It assumes that all atoms in the reactant molecule are atom-mapped and have
    corresponding atoms in the product molecule, and that the total number of atoms
    is conserved.
    """

    def __init__(
        self,
        smiles: str,
        label: float,
        featurizer_cfg: DictConfig,
        in_channel_multiplier: int = 2,
    ):
        super(CGR, self).__init__()

        self._atom_featurizer = hydra.utils.instantiate(
            featurizer_cfg.atom_featurizer_cfg
        )
        self._bond_featurizer = hydra.utils.instantiate(
            featurizer_cfg.bond_featurizer_cfg
        )

        smiles_reac, _, smiles_prod = smiles.split(">")

        self._mol_reac, _ = make_mol(smiles_reac)
        self._mol_prod, _ = make_mol(smiles_prod)

        self._ri2pi = map_reac_to_prod(self._mol_reac, self._mol_prod)

        n_atoms = self._mol_reac.GetNumAtoms()

        f_atoms_list: List[List[float]] = []
        atom_origin_type_list: List[AtomOriginType] = []

        for i in range(n_atoms):
            f_atoms_list.append(self._get_atom_features_internal(i))
            atom_origin_type_list.append(AtomOriginType.REACTANT_PRODUCT)

        edge_index_list: List[Tuple[int, int]] = []
        f_bonds_list: List[List[float]] = []
        edge_origin_type_list: List[EdgeOriginType] = []

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):  # Iterate over unique pairs
                bond_reac = self._mol_reac.GetBondBetweenAtoms(i, j)

                # Get corresponding product atom indices
                # These must exist due to checks in map_reac_to_prod
                idx_prod_i = self._ri2pi[i]
                idx_prod_j = self._ri2pi[j]
                bond_prod = self._mol_prod.GetBondBetweenAtoms(
                    idx_prod_i, idx_prod_j
                )

                if (
                    bond_reac is None and bond_prod is None
                ):  # No bond in either
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
        self.x = torch.tensor(f_atoms_list, dtype=torch.float)

        if edge_index_list:
            self.edge_index = (
                torch.tensor(edge_index_list, dtype=torch.long)
                .t()
                .contiguous()
            )
            self.edge_attr = torch.tensor(f_bonds_list, dtype=torch.float)
            self.edge_origin_type = torch.tensor(
                edge_origin_type_list, dtype=torch.long
            )
        else:  # Handle graphs with no edges
            self.edge_index = torch.empty((2, 0), dtype=torch.long)
            # Determine bond feature dimension for empty edge_attr
            dummy_bond_feat_len = len(
                self._get_bond_features_internal(None, None)
            )
            self.edge_attr = torch.empty(
                (0, dummy_bond_feat_len), dtype=torch.float
            )
            self.edge_origin_type = torch.empty((0), dtype=torch.long)

        self.y = torch.tensor([label], dtype=torch.float)
        self.smiles = smiles  # Store original reaction SMILES
        self.atom_origin_type = torch.tensor(
            atom_origin_type_list, dtype=torch.long
        )

        # num_nodes is a standard PyG attribute
        self.num_nodes = n_atoms

        # --- Clean up temporary construction attributes ---
        del self._mol_reac
        del self._mol_prod
        del self._ri2pi
        del self._atom_featurizer
        del self._bond_featurizer

    def _get_atom_features_internal(self, atom_idx: int) -> List[float]:
        """Helper to generate CGR atom features using temporary internal attributes."""
        f_atom_reac = self._atom_featurizer(
            self._mol_reac.GetAtomWithIdx(atom_idx)
        )
        # self._ri2pi[atom_idx] gives the corresponding product atom index
        f_atom_prod = self._atom_featurizer(
            self._mol_prod.GetAtomWithIdx(self._ri2pi[atom_idx])
        )
        f_atom_diff = [y - x for x, y in zip(f_atom_reac, f_atom_prod)]
        return f_atom_reac + f_atom_diff

    def _get_bond_features_internal(
        self, bond_reac: Optional[Chem.Bond], bond_prod: Optional[Chem.Bond]
    ) -> List[float]:
        """Helper to generate CGR bond features using temporary internal attributes."""
        # Assuming self._bond_featurizer(None) returns a list of default values (e.g., zeros)
        # of the correct base feature length. This is used to determine padding length.
        base_bond_feat_len = len(self._bond_featurizer(None))

        f_bond_reac = (
            self._bond_featurizer(bond_reac)
            if bond_reac
            else [0.0]
            * base_bond_feat_len  # Use float for consistency with PyTorch tensors
        )
        f_bond_prod = (
            self._bond_featurizer(bond_prod)
            if bond_prod
            else [0.0] * base_bond_feat_len
        )
        f_bond_diff = [y - x for x, y in zip(f_bond_reac, f_bond_prod)]
        return f_bond_reac + f_bond_diff
