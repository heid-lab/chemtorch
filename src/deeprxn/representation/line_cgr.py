from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch_geometric as tg
from rdkit import Chem

from deeprxn.representation.rxn_graph_base import AtomOriginType, RxnGraphBase


class LineCGR(RxnGraphBase):
    def __init__(
        self,
        smiles: str,
        label: float,
        atom_featurizer: callable,
        bond_featurizer: callable,
        in_channel_multiplier: int = 2,
        use_directed: bool = True,
        enthalpy=None,
    ):
        super().__init__(
            smiles=smiles,
            label=label,
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
            enthalpy=enthalpy,
        )

        self.n_atoms = self.mol_reac.GetNumAtoms()
        self.use_directed = use_directed

        # Compute feature lengths for zero vectors
        dummy_atom = None  # Dummy atom for feature length
        dummy_bond = None  # Dummy bond for feature length
        atom_feat_len = len(atom_featurizer(dummy_atom))
        bond_feat_len = len(bond_featurizer(dummy_bond))
        self.reactant_feat_length = (
            atom_feat_len + bond_feat_len + atom_feat_len
        )
        self.product_feat_length = (
            self.reactant_feat_length
        )  # Same featurizers

        # Existing graph construction
        self.line_nodes = []
        self.line_edges = []
        self.line_node_features = []
        self.line_edge_features = []
        self.atom_origin_type = []
        self.zero_reactant = [0.0] * self.reactant_feat_length
        self.zero_product = [0.0] * self.product_feat_length
        self._build_graph()

    def _build_graph(self):
        """Build line graph according to new approach."""
        cgr_edges = self._get_cgr_edges()
        reactant_line_features = self._get_reactant_line_features()
        product_line_features = self._get_product_line_features()

        # Create line nodes (CGR edges) and their features
        self.line_nodes = cgr_edges
        self.line_node_features = []
        for edge in cgr_edges:
            # Get reactant and product features, default to zeros if not present
            reactant_feat = reactant_line_features.get(
                edge, self.zero_reactant
            )
            product_feat = product_line_features.get(edge, self.zero_product)
            combined_feat = reactant_feat + product_feat
            self.line_node_features.append(combined_feat)

        # Create line edges (adjacency in line graph)
        self.line_edges, self.line_edge_features = self._create_line_edges(
            cgr_edges
        )

    def _get_cgr_edges(self) -> List[Union[Tuple[int, int], frozenset]]:
        """Collect all unique edges present in CGR (reactant OR product bonds)."""
        edges: Set[Union[Tuple[int, int], frozenset]] = set()
        for i in range(self.n_atoms):
            for j in range(self.n_atoms):
                if i == j:
                    continue

                bond_reac = self.mol_reac.GetBondBetweenAtoms(i, j)
                bond_prod = self.mol_prod.GetBondBetweenAtoms(
                    self.ri2pi[i], self.ri2pi[j]
                )

                if bond_reac or bond_prod:
                    if self.use_directed:
                        edges.add((i, j))
                    else:
                        edges.add(frozenset({i, j}))
        return list(edges)

    def _get_reactant_line_features(
        self,
    ) -> Dict[Union[Tuple[int, int], frozenset], List[float]]:
        """Generate features for reactant line graph nodes."""
        features = {}
        for bond in self.mol_reac.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            if self.use_directed:
                key = (i, j)
                features[key] = self._get_reactant_line_node_feature(i, j)
            else:
                key = frozenset({i, j})
                features[key] = self._get_reactant_line_node_feature(i, j)
        return features

    def _get_reactant_line_node_feature(self, i: int, j: int) -> List[float]:
        """Feature for reactant line graph node (i,j)."""
        atom_i_feat = self.atom_featurizer(self.mol_reac.GetAtomWithIdx(i))
        bond = self.mol_reac.GetBondBetweenAtoms(i, j)
        bond_feat = (
            self.bond_featurizer(bond)
            if bond
            else [0.0] * len(self.bond_featurizer(None))
        )
        atom_j_feat = self.atom_featurizer(self.mol_reac.GetAtomWithIdx(j))
        return atom_i_feat + bond_feat + atom_j_feat

    def _get_product_line_features(
        self,
    ) -> Dict[Union[Tuple[int, int], frozenset], List[float]]:
        """Generate features for product line graph nodes (mapped to reactant indices)."""
        features = {}
        pi2ri = {v: k for k, v in self.ri2pi.items()}
        for bond in self.mol_prod.GetBonds():
            pi = bond.GetBeginAtomIdx()
            pj = bond.GetEndAtomIdx()
            i = pi2ri[pi]
            j = pi2ri[pj]
            if self.use_directed:
                key = (i, j)
            else:
                key = frozenset({i, j})
            features[key] = self._get_product_line_node_feature(pi, pj)
        return features

    def _get_product_line_node_feature(self, pi: int, pj: int) -> List[float]:
        """Feature for product line graph node (pi,pj) mapped to (i,j)."""
        atom_pi_feat = self.atom_featurizer(self.mol_prod.GetAtomWithIdx(pi))
        bond = self.mol_prod.GetBondBetweenAtoms(pi, pj)
        bond_feat = (
            self.bond_featurizer(bond)
            if bond
            else [0.0] * len(self.bond_featurizer(None))
        )
        atom_pj_feat = self.atom_featurizer(self.mol_prod.GetAtomWithIdx(pj))
        return atom_pi_feat + bond_feat + atom_pj_feat

    def _create_line_edges(
        self, cgr_edges: List[Union[Tuple[int, int], frozenset]]
    ) -> Tuple[List[Tuple[int, int]], List[List[float]]]:
        """Create line graph edges between adjacent CGR edges."""
        edges = []
        features = []
        for idx_a, edge_a in enumerate(cgr_edges):
            for idx_b, edge_b in enumerate(cgr_edges):
                if idx_a == idx_b:
                    continue

                if self._edges_adjacent(edge_a, edge_b):
                    edges.append((idx_a, idx_b))
                    if not self.use_directed:
                        edges.append((idx_b, idx_a))
                    features.append([1.0])
        return edges, features

    def _edges_adjacent(
        self,
        edge_a: Union[Tuple[int, int], frozenset],
        edge_b: Union[Tuple[int, int], frozenset],
    ) -> bool:
        """Check if two CGR edges share a common atom."""
        if self.use_directed:
            # Directed: edge_a's dest matches edge_b's src
            return edge_a[1] == edge_b[0]
        else:
            # Undirected: any shared atom between edges
            a_set = set(edge_a)
            b_set = set(edge_b)
            return not a_set.isdisjoint(b_set)

    def to_pyg_data(self) -> tg.data.Data:
        """Convert to PyTorch Geometric Data object."""
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

        if self.enthalpy is not None:
            data.enthalpy = torch.tensor([self.enthalpy], dtype=torch.float)

        return data
