from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import hydra
import torch
import torch_geometric as tg
from omegaconf import DictConfig

from deeprxn.representation.rxn_graph import (
    AtomOriginType,
    EdgeOriginType,
    RxnGraphBase,
)


class UndirectedLineConnectedPairGraph(RxnGraphBase):
    """Line graph representation with separate reactant and product graphs."""

    def __init__(
        self,
        smiles: str,
        label: float,
        atom_featurizer: callable,
        bond_featurizer: callable,
        connection_direction: str = "bidirectional",
        concat_origin_feature: bool = False,
        in_channel_multiplier: int = 1,
        pre_transform_cfg: Optional[Dict[str, DictConfig]] = None,
    ):
        """

        Args:
        """
        super().__init__(
            smiles=smiles,
            label=label,
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
        )
        self.connection_direction = connection_direction
        self.pre_transform_cfg = pre_transform_cfg

        self.n_atoms_reac = self.mol_reac.GetNumAtoms()
        self.n_atoms_prod = self.mol_prod.GetNumAtoms()
        self.n_reactant_compounds = max(self.reac_origins) + 1

        self.zero_edge_features = [0.0] * len(self.bond_featurizer(None))

        self.component_features = {}
        self.original_to_enhanced_features = {}

        self.line_nodes = []
        self.line_edges = []
        self.line_node_features = []
        self.line_edge_features = []
        self.atom_origin_type = []
        self.edge_origin_type = []
        self.atom_compound_idx = []

        if self.pre_transform_cfg is not None:
            self._apply_component_transforms()

        self._build_graph()

    def _create_component_graph(
        self, mol, atom_indices, compound_idx
    ) -> tg.data.Data:
        """Create a graph for a single component."""
        data = tg.data.Data()

        x = []
        for idx in atom_indices:
            x.append(self.atom_featurizer(mol.GetAtomWithIdx(idx)))
        data.x = torch.tensor(x, dtype=torch.float)

        edges = []
        edge_features = []

        if len(atom_indices) > 1:
            global_to_local = {
                global_idx: local_idx
                for local_idx, global_idx in enumerate(atom_indices)
            }

            for i, global_i in enumerate(atom_indices):
                for j, global_j in enumerate(atom_indices[i + 1 :], i + 1):
                    bond = mol.GetBondBetweenAtoms(global_i, global_j)
                    if bond:
                        local_i = global_to_local[global_i]
                        local_j = global_to_local[global_j]
                        edges.extend([[local_i, local_j], [local_j, local_i]])
                        f_bond = self.bond_featurizer(bond)
                        edge_features.extend([f_bond, f_bond])

        data.edge_index = (
            torch.tensor(edges, dtype=torch.long).t().contiguous()
            if edges
            else torch.zeros((2, 0), dtype=torch.long)
        )
        data.edge_attr = (
            torch.tensor(edge_features, dtype=torch.float)
            if edge_features
            else torch.zeros(
                (0, len(self.bond_featurizer(None))), dtype=torch.float
            )
        )
        data.num_nodes = len(atom_indices)
        data.global_indices = torch.tensor(atom_indices, dtype=torch.long)
        data.compound_idx = compound_idx

        return data

    def _apply_component_transforms(self):
        """Apply transforms to each component and store enhanced atom features."""
        for _, config in self.pre_transform_cfg.items():
            transform = hydra.utils.instantiate(config)
            attr_names = transform.attr_name
            if isinstance(attr_names, str):
                attr_names = [attr_names]

            for compound_idx, indices in self._get_component_indices(
                self.reac_origins
            ).items():
                data = self._create_component_graph(
                    self.mol_reac, indices, compound_idx
                )
                data = transform(data)

                for local_idx, global_idx in enumerate(indices):
                    base_features = self.atom_featurizer(
                        self.mol_reac.GetAtomWithIdx(global_idx)
                    )
                    enhanced_features = base_features.copy()

                    for attr_name in attr_names:
                        if hasattr(data, attr_name):
                            feature_values = getattr(data, attr_name)[
                                local_idx
                            ].tolist()
                            enhanced_features.extend(feature_values)

                    self.original_to_enhanced_features[
                        (compound_idx, global_idx)
                    ] = enhanced_features

            for compound_idx, indices in self._get_component_indices(
                self.prod_origins
            ).items():
                data = self._create_component_graph(
                    self.mol_prod, indices, compound_idx
                )
                data = transform(data)

                for local_idx, global_idx in enumerate(indices):
                    base_features = self.atom_featurizer(
                        self.mol_prod.GetAtomWithIdx(global_idx)
                    )
                    enhanced_features = base_features.copy()

                    for attr_name in attr_names:
                        if hasattr(data, attr_name):
                            feature_values = getattr(data, attr_name)[
                                local_idx
                            ].tolist()
                            enhanced_features.extend(feature_values)

                    product_compound_idx = (
                        compound_idx + self.n_reactant_compounds
                    )
                    self.original_to_enhanced_features[
                        (product_compound_idx, global_idx)
                    ] = enhanced_features

    def _get_enhanced_atom_features(self, mol, atom_idx, compound_idx):
        """Get atom features with pre-transform features if available."""
        if self.pre_transform_cfg is not None:
            return self.original_to_enhanced_features.get(
                (compound_idx, atom_idx),
                self.atom_featurizer(mol.GetAtomWithIdx(atom_idx)),
            )
        return self.atom_featurizer(mol.GetAtomWithIdx(atom_idx))

    def _build_reactant_line_graph(self):
        """Build line graph for reactant molecules."""
        edge_map = {}
        current_idx = 0

        atoms_with_bonds = set()

        processed_edges = set()

        for i in range(self.n_atoms_reac):
            for j in range(self.n_atoms_reac):
                if i != j:
                    edge_pair = tuple(sorted([i, j]))  # undirected
                    if edge_pair not in processed_edges:
                        bond = self.mol_reac.GetBondBetweenAtoms(i, j)
                        if bond:
                            compound_i = self.reac_origins[i]
                            compound_j = self.reac_origins[j]
                            if compound_i == compound_j:
                                atom_i_features = (
                                    self._get_enhanced_atom_features(
                                        self.mol_reac, i, compound_i
                                    )
                                )
                                atom_j_features = (
                                    self._get_enhanced_atom_features(
                                        self.mol_reac, j, compound_i
                                    )
                                )
                                bond_features = self.bond_featurizer(bond)

                                node_features = (
                                    atom_i_features
                                    + bond_features
                                    + atom_j_features
                                )

                                self.line_nodes.append((i, j))
                                self.line_node_features.append(node_features)
                                self.atom_origin_type.append(
                                    AtomOriginType.REACTANT
                                )
                                self.atom_compound_idx.append(compound_i)
                                edge_map[(i, j)] = current_idx
                                edge_map[(j, i)] = current_idx
                                current_idx += 1
                                atoms_with_bonds.add(i)
                                atoms_with_bonds.add(j)
                                processed_edges.add(edge_pair)

        for i in range(self.n_atoms_reac):
            if i not in atoms_with_bonds:
                compound_i = self.reac_origins[i]
                atom_features = self._get_enhanced_atom_features(
                    self.mol_reac, i, compound_i
                )
                node_features = (
                    atom_features + self.zero_edge_features + atom_features
                )

                self.line_nodes.append((i, i))  # Self-loop in original graph
                self.line_node_features.append(node_features)
                self.atom_origin_type.append(AtomOriginType.REACTANT)
                self.atom_compound_idx.append(self.reac_origins[i])
                edge_map[(i, i)] = current_idx
                current_idx += 1

        processed_edge_pairs = set()
        for (src1, tgt1), idx1 in edge_map.items():
            for (src2, tgt2), idx2 in edge_map.items():
                if idx1 != idx2 and (idx1, idx2) not in processed_edge_pairs:
                    # Check if edges share any vertex
                    if (
                        src1 == src2
                        or src1 == tgt2
                        or tgt1 == src2
                        or tgt1 == tgt2
                    ):
                        if (
                            self.atom_compound_idx[idx1]
                            == self.atom_compound_idx[idx2]
                        ):
                            self.line_edges.append((idx1, idx2))
                            self.line_edges.append(
                                (idx2, idx1)
                            )  # Add both directions
                            self.line_edge_features.extend(
                                [[1.0, 0.0], [1.0, 0.0]]
                            )
                            self.edge_origin_type.extend(
                                [
                                    EdgeOriginType.REACTANT,
                                    EdgeOriginType.REACTANT,
                                ]
                            )
                            processed_edge_pairs.add((idx1, idx2))
                            processed_edge_pairs.add((idx2, idx1))

    @staticmethod
    def _get_component_indices(origins) -> Dict[int, List[int]]:
        """Get atom indices for each component in a molecule.

        Args:
            origins: List of component indices for each atom

        Returns:
            Dictionary mapping component index to list of atom indices in that component
        """
        component_indices = {}
        for atom_idx, origin in enumerate(origins):
            if origin not in component_indices:
                component_indices[origin] = []
            component_indices[origin].append(atom_idx)
        return component_indices

    def _build_product_line_graph(self):
        """Build line graph for product molecules."""
        offset = len(self.line_nodes)
        edge_map = {}
        current_idx = offset

        atoms_with_bonds = set()
        processed_edges = set()

        for i in range(self.n_atoms_prod):
            prod_i = self.ri2pi[i]
            for j in range(self.n_atoms_prod):
                if i != j:
                    prod_j = self.ri2pi[j]
                    edge_pair = tuple(sorted([i, j]))
                    if edge_pair not in processed_edges:
                        bond = self.mol_prod.GetBondBetweenAtoms(
                            prod_i, prod_j
                        )
                        if bond:
                            compound_i = self.prod_origins[prod_i]
                            compound_j = self.prod_origins[prod_j]
                            if compound_i == compound_j:
                                product_compound_idx = (
                                    compound_i + self.n_reactant_compounds
                                )
                                atom_i_features = (
                                    self._get_enhanced_atom_features(
                                        self.mol_prod,
                                        prod_i,
                                        product_compound_idx,
                                    )
                                )
                                atom_j_features = (
                                    self._get_enhanced_atom_features(
                                        self.mol_prod,
                                        prod_j,
                                        product_compound_idx,
                                    )
                                )
                                bond_features = self.bond_featurizer(bond)

                                node_features = (
                                    atom_i_features
                                    + bond_features
                                    + atom_j_features
                                )

                                self.line_nodes.append((i, j))
                                self.line_node_features.append(node_features)
                                self.atom_origin_type.append(
                                    AtomOriginType.PRODUCT
                                )
                                self.atom_compound_idx.append(
                                    compound_i + self.n_reactant_compounds
                                )
                                edge_map[(i, j)] = current_idx
                                edge_map[(j, i)] = current_idx
                                current_idx += 1
                                atoms_with_bonds.add(i)
                                atoms_with_bonds.add(j)
                                processed_edges.add(edge_pair)

        for i in range(self.n_atoms_prod):
            if i not in atoms_with_bonds:
                prod_i = self.ri2pi[i]
                compound_i = self.prod_origins[prod_i]
                product_compound_idx = compound_i + self.n_reactant_compounds

                atom_features = self._get_enhanced_atom_features(
                    self.mol_prod, prod_i, product_compound_idx
                )
                node_features = (
                    atom_features + self.zero_edge_features + atom_features
                )

                self.line_nodes.append((i, i))
                self.line_node_features.append(node_features)
                self.atom_origin_type.append(AtomOriginType.PRODUCT)
                self.atom_compound_idx.append(
                    self.prod_origins[prod_i] + self.n_reactant_compounds
                )
                edge_map[(i, i)] = current_idx
                current_idx += 1

        processed_edge_pairs = set()
        for (src1, tgt1), idx1 in edge_map.items():
            for (src2, tgt2), idx2 in edge_map.items():
                if idx1 != idx2 and (idx1, idx2) not in processed_edge_pairs:
                    # Check if edges share any vertex
                    if (
                        src1 == src2
                        or src1 == tgt2
                        or tgt1 == src2
                        or tgt1 == tgt2
                    ):
                        if (
                            self.atom_compound_idx[idx1]
                            == self.atom_compound_idx[idx2]
                        ):
                            self.line_edges.append((idx1, idx2))
                            self.line_edges.append(
                                (idx2, idx1)
                            )  # Add both directions
                            self.line_edge_features.extend(
                                [[0.0, 1.0], [0.0, 1.0]]
                            )
                            self.edge_origin_type.extend(
                                [
                                    EdgeOriginType.PRODUCT,
                                    EdgeOriginType.PRODUCT,
                                ]
                            )
                            processed_edge_pairs.add((idx1, idx2))
                            processed_edge_pairs.add((idx2, idx1))

    def _build_graph(self):
        """Build line graph representation."""
        self._build_reactant_line_graph()
        self._build_product_line_graph()
        # Connection logic between reactant and product line graphs will be added later

    def to_pyg_data(self) -> tg.data.Data:
        """Convert the line graph to a PyTorch Geometric Data object."""
        data = tg.data.Data()
        data.x = torch.tensor(self.line_node_features, dtype=torch.float)
        data.edge_index = (
            torch.tensor(self.line_edges, dtype=torch.long).t().contiguous()
        )
        if self.line_edge_features:
            data.edge_attr = torch.tensor(
                self.line_edge_features, dtype=torch.float
            )
        data.y = torch.tensor([self.label], dtype=torch.float)
        data.smiles = self.smiles
        data.atom_origin_type = torch.tensor(
            self.atom_origin_type, dtype=torch.long
        )
        data.edge_origin_type = torch.tensor(
            self.edge_origin_type, dtype=torch.long
        )
        data.atom_compound_idx = torch.tensor(
            self.atom_compound_idx, dtype=torch.long
        )

        return data
