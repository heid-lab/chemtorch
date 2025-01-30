from typing import Optional

import hydra
import torch
import torch_geometric as tg
from omegaconf import DictConfig
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
        concat_transform_features: bool = False,
        pre_transform_cfg: Optional[DictConfig] = None,
        enthalpy=None,
    ):
        super().__init__(
            smiles=smiles,
            label=label,
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
            enthalpy=enthalpy,
        )

        if pre_transform_cfg is not None and use_directed:
            raise ValueError(
                "Pre-transforms for LineCGR are only supported when use_directed is False."
            )

        self.n_atoms = self.mol_reac.GetNumAtoms()
        self.use_directed = use_directed
        self.pre_transform_cfg = pre_transform_cfg
        self.concat_transform_features = concat_transform_features
        self.component_features = {}  # Maps edges to features from components
        self.merged_transform_features = {}

        # Existing graph construction
        self.line_nodes = []
        self.line_edges = []
        self.line_node_features = []
        self.line_edge_features = []
        self.atom_origin_type = []
        self.zero_bond_features = [0] * len(self.bond_featurizer(None))
        self._build_graph()

        # Apply pre-transforms if applicable
        if not self.use_directed and self.pre_transform_cfg is not None:
            self._apply_component_transforms()
            self._merge_transform_features()

    def _build_graph(self):
        """Build line graph according to graph theory principles."""
        cgr_edges = self._get_cgr_edges()

        self.line_nodes, self.line_node_features = self._create_line_nodes(
            cgr_edges
        )

        self.line_edges, self.line_edge_features = self._create_line_edges(
            cgr_edges
        )

    def _get_cgr_edges(self) -> list[tuple[int, int]]:
        """Collect all unique edges present in CGR (reactant OR product bonds)."""
        edges = set()
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

    def _create_line_nodes(self, cgr_edges: list[tuple[int, int]]):
        """Create line graph nodes with proper features."""
        nodes = []
        features = []
        for edge in cgr_edges:
            i, j = edge if self.use_directed else tuple(edge)
            bond_reac = self.mol_reac.GetBondBetweenAtoms(i, j)
            bond_prod = self.mol_prod.GetBondBetweenAtoms(
                self.ri2pi[i], self.ri2pi[j]
            )

            # Feature: Bond diffs + connected atom diffs
            bond_features = self._get_bond_features(bond_reac, bond_prod)
            atom_i_features = self._get_atom_features(i)
            atom_j_features = self._get_atom_features(j)

            node_features = atom_i_features + bond_features + atom_j_features

            nodes.append((i, j))
            features.append(node_features)
            self.atom_origin_type.append(AtomOriginType.REACTANT_PRODUCT)

        return nodes, features

    def _create_line_edges(self, cgr_edges: list[tuple[int, int]]):
        """Create line graph edges between adjacent CGR edges."""
        edges = []
        features = []
        # edge_indices = {edge: idx for idx, edge in enumerate(cgr_edges)}

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

    def _edges_adjacent(self, edge_a, edge_b) -> bool:
        """Check if two CGR edges share a common atom."""
        if self.use_directed:
            # Directed: edge_a's dest matches edge_b's src
            return edge_a[1] == edge_b[0]
        else:
            # Undirected: any shared atom between edges
            a_set = set(edge_a)
            b_set = set(edge_b)
            return not a_set.isdisjoint(b_set)

    def _get_atom_features(self, atom_idx: int) -> list[float]:
        f_atom_reac = self.atom_featurizer(
            self.mol_reac.GetAtomWithIdx(atom_idx)
        )
        f_atom_prod = self.atom_featurizer(
            self.mol_prod.GetAtomWithIdx(self.ri2pi[atom_idx])
        )
        f_atom_diff = [y - x for x, y in zip(f_atom_reac, f_atom_prod)]
        return f_atom_reac + f_atom_diff

    def _get_bond_features(
        self, bond_reac: Optional[Chem.Bond], bond_prod: Optional[Chem.Bond]
    ) -> list[float]:
        f_bond_reac = (
            self.bond_featurizer(bond_reac)
            if bond_reac
            else self.zero_bond_features
        )
        f_bond_prod = (
            self.bond_featurizer(bond_prod)
            if bond_prod
            else self.zero_bond_features
        )
        f_bond_diff = [y - x for x, y in zip(f_bond_reac, f_bond_prod)]
        return f_bond_reac + f_bond_diff

    def _apply_component_transforms(self):
        """Apply transforms to each component's line graph."""
        for _, config in self.pre_transform_cfg.items():
            transform = hydra.utils.instantiate(config)
            attr_names = transform.attr_name
            if isinstance(attr_names, str):
                attr_names = [attr_names]
            for attr_name in attr_names:
                self.merged_transform_features[attr_name] = None
            # Process reactant components
            reac_components = self._get_component_indices(self.reac_origins)
            for compound_idx, atom_indices in reac_components.items():
                component_edges = self._get_component_edges(
                    self.mol_reac, atom_indices
                )
                if not component_edges:
                    continue
                line_graph_data = self._create_component_line_graph(
                    component_edges
                )
                transformed_data = transform(line_graph_data)
                self._store_component_features(
                    transformed_data, component_edges, attr_names, "reactant"
                )

            # Process product components
            prod_components = self._get_component_indices(self.prod_origins)
            for compound_idx, atom_indices in prod_components.items():
                component_edges = self._get_component_edges(
                    self.mol_prod, atom_indices
                )
                if not component_edges:
                    continue
                line_graph_data = self._create_component_line_graph(
                    component_edges
                )
                transformed_data = transform(line_graph_data)
                self._store_component_features(
                    transformed_data, component_edges, attr_names, "product"
                )

    def _store_component_features(
        self, transformed_data, component_edges, attr_names, origin
    ):
        """Store features from transformed component line graph."""
        for attr_name in attr_names:
            if hasattr(transformed_data, attr_name):
                features = getattr(transformed_data, attr_name)
                for edge_idx, edge in enumerate(component_edges):
                    if edge not in self.component_features:
                        self.component_features[edge] = {
                            "reactant": {},
                            "product": {},
                        }
                    self.component_features[edge][origin][attr_name] = (
                        features[edge_idx]
                    )

    def _merge_transform_features(self):
        """Merge component features into LineCGR's line nodes."""
        for attr_name in self.merged_transform_features.keys():
            # Determine the feature size for this attribute
            feat_size = None
            # Iterate through component features to find the first occurrence of attr_name
            for edge in self.component_features:
                edge_data = self.component_features[edge]
                for origin in ["reactant", "product"]:
                    if origin in edge_data and attr_name in edge_data[origin]:
                        feat = edge_data[origin][attr_name]
                        # Get feature size excluding batch dimensions (assume last dimension is feature)
                        feat_size = feat.size()[-1]
                        break
                if feat_size is not None:
                    break
            # If no features found, default to 1 (but this should be handled if transforms are applied)
            if feat_size is None:
                feat_size = 1

            features = []
            for line_node in self.line_nodes:
                edge = (
                    tuple(sorted(line_node))
                    if not self.use_directed
                    else line_node
                )
                # Get reactant and product features with default matching feature size
                reactant_feat = (
                    self.component_features.get(edge, {})
                    .get("reactant", {})
                    .get(attr_name, torch.zeros(feat_size))
                )
                product_feat = (
                    self.component_features.get(edge, {})
                    .get("product", {})
                    .get(attr_name, torch.zeros(feat_size))
                )
                # Ensure features are at least 1D tensors for concatenation
                if reactant_feat.dim() == 0:
                    reactant_feat = reactant_feat.unsqueeze(-1)
                if product_feat.dim() == 0:
                    product_feat = product_feat.unsqueeze(-1)
                merged_feat = torch.cat([reactant_feat, product_feat], dim=-1)
                features.append(merged_feat)
            # Stack all features which now have consistent size
            self.merged_transform_features[attr_name] = (
                torch.stack(features) if features else torch.tensor([])
            )

    @staticmethod
    def _get_component_indices(origins) -> dict[int, list[int]]:
        """Group atoms by component origin."""
        component_indices = {}
        for atom_idx, origin in enumerate(origins):
            component_indices.setdefault(origin, []).append(atom_idx)
        return component_indices

    def _get_component_edges(self, mol, atom_indices):
        """Collect edges (bonds) of a component as sorted tuples."""
        edges = []
        for bond in mol.GetBonds():
            begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if begin in atom_indices and end in atom_indices:
                edges.append(tuple(sorted((begin, end))))
        return list(set(edges))

    def _create_component_line_graph(self, component_edges):
        """Create a line graph for the component's edges."""
        edge_index = []
        for i in range(len(component_edges)):
            for j in range(len(component_edges)):
                if i != j and self._edges_adjacent(
                    component_edges[i], component_edges[j]
                ):
                    edge_index.extend([(i, j), (j, i)])
        return tg.data.Data(
            edge_index=(
                torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                if edge_index
                else torch.empty((2, 0), dtype=torch.long)
            ),
            num_nodes=len(component_edges),
        )

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

        if self.merged_transform_features:
            if self.concat_transform_features:
                data.x = torch.cat(
                    [data.x]
                    + [
                        feat
                        for feat in self.merged_transform_features.values()
                    ],
                    dim=1,
                )
            else:
                for name, feat in self.merged_transform_features.items():
                    setattr(data, name, feat)

        if self.enthalpy is not None:
            data.enthalpy = torch.tensor([self.enthalpy], dtype=torch.float)

        return data
