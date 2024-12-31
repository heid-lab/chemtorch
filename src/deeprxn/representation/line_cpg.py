from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
import torch_geometric as tg
from omegaconf import DictConfig

from deeprxn.representation.rxn_graph import (
    AtomOriginType,
    EdgeOriginType,
    RxnGraphBase,
)


class LineConnectedPairGraph(RxnGraphBase):
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
        """Initialize line graph representation.

        Args:
            smiles: SMARTS reaction string with atom mapping
            label: Reaction label
            atom_featurizer: Function to generate atom features
            bond_featurizer: Function to generate bond features
            connection_direction: How to connect corresponding edges:
                None: No connections
                "bidirectional": Both directions
                "reactants_to_products": Reactant to product only
                "products_to_reactants": Product to reactant only
        """
        super().__init__(
            smiles=smiles,
            label=label,
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
        )
        self.connection_direction = connection_direction
        self.pre_transform_cfg = pre_transform_cfg

        # Track original graph information
        self.n_atoms_reac = self.mol_reac.GetNumAtoms()
        self.n_atoms_prod = self.mol_prod.GetNumAtoms()
        self.n_reactant_compounds = (
            max(self.reac_origins) + 1
        )  # Track number of reactant compounds

        self.zero_edge_features = [0.0] * len(self.bond_featurizer(None))

        # Line graph specific storage
        self.line_nodes = []  # Original edges become nodes
        self.line_edges = []  # New edges between adjacent bonds
        self.line_node_features = (
            []
        )  # Original bond features become node features
        self.line_edge_features = []  # New features for line graph edges
        self.atom_origin_type = (
            []
        )  # Track origin of line graph nodes (original edges)
        self.edge_origin_type = []  # Track origin of line graph edges
        self.atom_compound_idx = (
            []
        )  # Track compound index for each line graph node

        self._build_graph()

    def _build_reactant_line_graph(self):
        """Build line graph for reactant molecules."""
        edge_map = {}  # Maps (source_idx, target_idx) to line graph node index
        current_idx = 0

        # Track atoms that have bonds
        atoms_with_bonds = set()

        # Create nodes from directed edges (bonds)
        for i in range(self.n_atoms_reac):
            for j in range(self.n_atoms_reac):
                if i != j:
                    bond = self.mol_reac.GetBondBetweenAtoms(i, j)
                    if bond:
                        compound_i = self.reac_origins[i]
                        compound_j = self.reac_origins[j]
                        if compound_i == compound_j:
                            source_atom_features = self.atom_featurizer(
                                self.mol_reac.GetAtomWithIdx(i)
                            )
                            bond_features = self.bond_featurizer(bond)

                            node_features = (
                                source_atom_features + bond_features
                            )

                            self.line_nodes.append((i, j))
                            self.line_node_features.append(node_features)
                            self.atom_origin_type.append(
                                AtomOriginType.REACTANT
                            )
                            self.atom_compound_idx.append(compound_i)
                            edge_map[(i, j)] = current_idx
                            current_idx += 1
                            atoms_with_bonds.add(i)

        # Create nodes for isolated atoms
        for i in range(self.n_atoms_reac):
            if i not in atoms_with_bonds:
                # Create node features from atom and zero edge features
                atom_features = self.atom_featurizer(
                    self.mol_reac.GetAtomWithIdx(i)
                )
                node_features = atom_features + self.zero_edge_features

                self.line_nodes.append((i, i))  # Self-loop in original graph
                self.line_node_features.append(node_features)
                self.atom_origin_type.append(AtomOriginType.REACTANT)
                self.atom_compound_idx.append(self.reac_origins[i])
                edge_map[(i, i)] = current_idx
                current_idx += 1

        # Create edges between adjacent bonds
        for (src1, tgt1), idx1 in edge_map.items():
            for (src2, tgt2), idx2 in edge_map.items():
                if (
                    tgt1 == src2 and idx1 != idx2
                ):  # First edge's target is second edge's source
                    if (
                        self.atom_compound_idx[idx1]
                        == self.atom_compound_idx[idx2]
                    ):
                        self.line_edges.append((idx1, idx2))
                        self.line_edge_features.append([1.0, 0.0])
                        self.edge_origin_type.append(EdgeOriginType.REACTANT)

    def _build_product_line_graph(self):
        """Build line graph for product molecules."""
        offset = len(self.line_nodes)
        edge_map = {}
        current_idx = offset

        # Track atoms that have bonds
        atoms_with_bonds = set()

        # Create nodes from directed edges (bonds)
        for i in range(self.n_atoms_prod):
            prod_i = self.ri2pi[i]
            for j in range(self.n_atoms_prod):
                if i != j:
                    prod_j = self.ri2pi[j]
                    bond = self.mol_prod.GetBondBetweenAtoms(prod_i, prod_j)
                    if bond:
                        compound_i = self.prod_origins[prod_i]
                        compound_j = self.prod_origins[prod_j]
                        if compound_i == compound_j:
                            source_atom_features = self.atom_featurizer(
                                self.mol_prod.GetAtomWithIdx(prod_i)
                            )
                            bond_features = self.bond_featurizer(bond)

                            node_features = (
                                source_atom_features + bond_features
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
                            current_idx += 1
                            atoms_with_bonds.add(i)

        # Create nodes for isolated atoms
        for i in range(self.n_atoms_prod):
            if i not in atoms_with_bonds:
                prod_i = self.ri2pi[i]
                # Create node features from atom and zero edge features
                atom_features = self.atom_featurizer(
                    self.mol_prod.GetAtomWithIdx(prod_i)
                )
                node_features = atom_features + self.zero_edge_features

                self.line_nodes.append((i, i))  # Self-loop in original graph
                self.line_node_features.append(node_features)
                self.atom_origin_type.append(AtomOriginType.PRODUCT)
                self.atom_compound_idx.append(
                    self.prod_origins[prod_i] + self.n_reactant_compounds
                )
                edge_map[(i, i)] = current_idx
                current_idx += 1

        # Create edges between adjacent bonds
        for (src1, tgt1), idx1 in edge_map.items():
            for (src2, tgt2), idx2 in edge_map.items():
                if (
                    tgt1 == src2 and idx1 != idx2
                ):  # First edge's target is second edge's source
                    if (
                        self.atom_compound_idx[idx1]
                        == self.atom_compound_idx[idx2]
                    ):
                        self.line_edges.append((idx1, idx2))
                        self.line_edge_features.append([0.0, 1.0])
                        self.edge_origin_type.append(EdgeOriginType.PRODUCT)

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
