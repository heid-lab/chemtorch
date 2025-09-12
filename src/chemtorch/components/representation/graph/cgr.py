from typing import List, Tuple
try:
    # Python ≥ 3.12
    from typing import override  # type: ignore
except ImportError:
    # Python < 3.12
    from typing_extensions import override  # type: ignore

import torch

from torch_geometric.data import Data
from rdkit.Chem import Atom, Bond

from chemtorch.components.representation.graph.featurizer.featurizer_base import FeaturizerBase
from chemtorch.components.representation.graph.featurizer.featurizer_compose import FeaturizerCompose
from chemtorch.components.representation.abstract_representation import AbstractRepresentation
from chemtorch.utils.atom_mapping import (
    AtomOriginType,
    EdgeOriginType,
    make_mol,
    map_reac_to_prod,
)


class CGR(AbstractRepresentation[Data]):
    """
    Stateless class for constructing Condensed Graph of Reaction (CGR) representations.

    This class does not hold any data itself. Instead, it provides a `forward()` method
    that takes a sample (e.g., a dict or pd.Series) and returns a PyTorch Geometric Data object
    representing the reaction as a graph.

    # TODO: Update docstring once featurizers are passed explicitly
    Usage:
        >>> cgr = CGR(featurizer_cfg)
        >>> data = cgr.construct(sample)
        >>> data = cgr(sample)  # equivalent to above line
    """

    def __init__(
        self,
        atom_featurizer: FeaturizerBase[Atom] | FeaturizerCompose,
        bond_featurizer: FeaturizerBase[Bond] | FeaturizerCompose,
        **kwargs,  # ignored, TODO: remove once all featurizers are passed explicitly
    ):
        """
        Initialize the CGR representation with atom and bond featurizers.

        Args:
            atom_featurizer (FeaturizerBase[Atom] | FeaturizerCompose):
                A featurizer for atom features, which can be a single featurizer or a composed one.
            bond_featurizer (FeaturizerBase[Bond] | FeaturizerCompose):
                A featurizer for bond features, which can also be a single featurizer or a composed one.
        """
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer

    @override
    def construct(self, smiles: str) -> Data:
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
            atom_reac = mol_reac.GetAtomWithIdx(i)
            atom_prod = mol_prod.GetAtomWithIdx(ri2pi[i])
            f_atom = self._compute_feature_and_diff(
                self.atom_featurizer, atom_reac, atom_prod
            )
            f_atoms_list.append(f_atom)
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

                f_bond = self._compute_feature_and_diff(
                    self.bond_featurizer, bond_reac, bond_prod
                )

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
                torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            )
            data.edge_attr = torch.tensor(f_bonds_list, dtype=torch.float)
            data.edge_origin_type = torch.tensor(
                edge_origin_type_list, dtype=torch.long
            )
        else:  # Handle graphs with no edges
            data.edge_index = torch.empty((2, 0), dtype=torch.long)
            # Determine bond feature dimension for empty edge_attr
            dummy_bond_feat_len = len(
                self._compute_feature_and_diff(self.bond_featurizer, None, None)
            )
            data.edge_attr = torch.empty((0, dummy_bond_feat_len), dtype=torch.float)
            data.edge_origin_type = torch.empty((0), dtype=torch.long)

        data.smiles = smiles  # Store original reaction SMILES
        data.atom_origin_type = torch.tensor(atom_origin_type_list, dtype=torch.long)

        # num_nodes is a standard PyG attribute
        data.num_nodes = n_atoms

        return data

    def _compute_feature_and_diff(self, featurizer, obj1, obj2) -> List[float]:
        """
        General helper to compute features and their difference for two objects (atom or bond).

        Args:
            featurizer: Callable that computes features for the object.
            obj1: First object (e.g., atom or bond).
            obj2: Second object (e.g., atom or bond).

        Returns:
            List[float]: Features for obj1 concatenated with the difference (obj2 - obj1).
        """
        f1 = featurizer(obj1)
        f2 = featurizer(obj2)
        f_diff = [y - x for x, y in zip(f1, f2)]
        return f1 + f_diff
