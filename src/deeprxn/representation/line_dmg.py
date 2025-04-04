import itertools
from typing import Dict, List, Optional

import hydra
import torch
import torch_geometric as tg
from omegaconf import DictConfig

from deeprxn.representation.rxn_graph_base import (
    AtomOriginType,
    EdgeOriginType,
    RxnGraphBase,
)


class LineDMG(RxnGraphBase):
    """Line graph representation of Dual Molecular Graph (DMG)."""

    def __init__(
        self,
        smiles: str,
        label: float,
        atom_featurizer: callable,
        bond_featurizer: callable,
        qm_featurizer= None,   
        single_featurizer= None,
        concat_origin_feature: bool = False,
        in_channel_multiplier: int = 1,
        pre_transform_cfg: Optional[Dict[str, DictConfig]] = None,
        use_directed: bool = True,
        feature_aggregation: Optional[str] = None,
        enthalpy=None,
    ):
        """

        Args:
        """
        super().__init__(
            smiles=smiles,
            label=label,
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
            enthalpy=enthalpy,
        )
        self.pre_transform_cfg = pre_transform_cfg
        self.use_directed = use_directed
        self.feature_aggregation = feature_aggregation

        dummy_atom = None
        dummy_bond = None
        atom_feat_len = len(atom_featurizer(dummy_atom))
        bond_feat_len = len(bond_featurizer(dummy_bond))
        
        if self.use_directed:
            self.reactant_feat_length = atom_feat_len + bond_feat_len
        else:
            if self.feature_aggregation in ["sum", "mean", "max"]:
                self.reactant_feat_length = atom_feat_len + bond_feat_len
            else:
                self.reactant_feat_length = atom_feat_len * 2 + bond_feat_len
        self.product_feat_length = self.reactant_feat_length

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
    
    def _add_self_loops(self, is_reactant: bool):
        """Add self-loops for isolated atoms with direction-aware features."""
        mol = self.mol_reac if is_reactant else self.mol_prod
        n_atoms = self.n_atoms_reac if is_reactant else self.n_atoms_prod
        origins = self.reac_origins if is_reactant else self.prod_origins
        compound_offset = 0 if is_reactant else self.n_reactant_compounds
        origin_type = AtomOriginType.REACTANT if is_reactant else AtomOriginType.PRODUCT

        for atom_idx in range(n_atoms):
            # Get global index for products
            if not is_reactant:
                global_idx = self.ri2pi[atom_idx]
            else:
                global_idx = atom_idx

            # Check if atom has any bonds
            has_bonds = any(mol.GetBondBetweenAtoms(global_idx, j) 
                        for j in range(mol.GetNumAtoms()) if j != global_idx)

            if not has_bonds:
                compound_idx = origins[global_idx] + compound_offset
                atom_feat = self._get_enhanced_atom_features(mol, global_idx, compound_idx)
                
                if self.use_directed:
                    features = atom_feat + self.zero_edge_features
                else:
                    if self.feature_aggregation in ["sum", "mean", "max"]:
                        features = atom_feat + self.zero_edge_features
                    else:
                        features = atom_feat + atom_feat + self.zero_edge_features

                self.line_nodes.append((atom_idx, atom_idx))
                self.line_node_features.append(features)
                self.atom_origin_type.append(origin_type)
                self.atom_compound_idx.append(compound_idx)
    
    def _get_line_node_features(self, atom1_feat: List[float], 
                               atom2_feat: List[float],
                               bond_feat: List[float]) -> List[float]:
        """Create line node features based on directionality and aggregation."""
        if self.use_directed:
            # Directed: source atom + bond features
            return atom1_feat + bond_feat
        else:
            # Undirected: apply aggregation if specified
            if self.feature_aggregation == "sum":
                return [a+b for a,b in zip(atom1_feat, atom2_feat)] + bond_feat
            elif self.feature_aggregation == "mean":
                return [(a+b)/2 for a,b in zip(atom1_feat, atom2_feat)] + bond_feat
            elif self.feature_aggregation == "max":
                return [max(a,b) for a,b in zip(atom1_feat, atom2_feat)] + bond_feat
            else:  # Concatenation
                return atom1_feat + bond_feat + atom2_feat

    def _build_reactant_line_graph(self):
        """Build line graph for reactant molecules with undirected edges."""
        edge_map = {}  # Maps sorted edge tuples to line node indices
        current_idx = 0
        processed_edges = set()  # Track undirected edges

        # Step 1: Create line nodes for reactant edges
        for i in range(self.n_atoms_reac):
            for j in range(i + 1, self.n_atoms_reac):  # Avoid duplicate checks
                bond = self.mol_reac.GetBondBetweenAtoms(i, j)
                if bond:
                    compound_i = self.reac_origins[i]
                    compound_j = self.reac_origins[j]
                    if compound_i != compound_j:
                        continue

                    # Use sorted tuple for undirected edges
                    edge = tuple(sorted((i, j)))
                    if edge in processed_edges:
                        continue
                    processed_edges.add(edge)

                    # Get features
                    atom_i_feat = self._get_enhanced_atom_features(
                        self.mol_reac, i, compound_i
                    )
                    atom_j_feat = self._get_enhanced_atom_features(
                        self.mol_reac, j, compound_j
                    )
                    bond_feat = self.bond_featurizer(bond)
                    node_features = self._get_line_node_features(
                        atom_i_feat, atom_j_feat, bond_feat
                    )

                    # Add line node (both directions map to same index)
                    self.line_nodes.append(edge)
                    self.line_node_features.append(node_features)
                    self.atom_origin_type.append(AtomOriginType.REACTANT)
                    self.atom_compound_idx.append(compound_i)
                    edge_map[edge] = current_idx
                    current_idx += 1

        # Step 2: Add self-loops for isolated atoms (same as before)
        self._add_self_loops(is_reactant=True)

        # Step 3: Create line edges (adjacency)
        processed_pairs = set()
        edges = list(edge_map.items())
        
        for (edge_a, idx1), (edge_b, idx2) in itertools.product(edges, edges):
            if idx1 == idx2:
                continue

            # Check adjacency (shared atom in original graph)
            a_nodes = set(edge_a)
            b_nodes = set(edge_b)
            if a_nodes.intersection(b_nodes):
                # Ensure we process each pair only once
                pair = tuple(sorted((idx1, idx2)))
                if pair not in processed_pairs:
                    self.line_edges.extend([(idx1, idx2), (idx2, idx1)])
                    self.line_edge_features.extend([[1.0, 0.0], [1.0, 0.0]])
                    self.edge_origin_type.extend(
                        [EdgeOriginType.REACTANT, EdgeOriginType.REACTANT]
                    )
                    processed_pairs.add(pair)

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
        """Build line graph for product molecules with undirected edges."""
        offset = len(self.line_nodes)
        edge_map = {}
        current_idx = offset
        processed_edges = set()

        # Step 1: Create line nodes for product edges (analogous to reactants)
        for i in range(self.n_atoms_prod):
            pi = self.ri2pi[i]
            for j in range(i + 1, self.n_atoms_prod):
                pj = self.ri2pi[j]
                bond = self.mol_prod.GetBondBetweenAtoms(pi, pj)
                if bond:
                    compound_pi = self.prod_origins[pi]
                    compound_pj = self.prod_origins[pj]
                    if compound_pi != compound_pj:
                        continue

                    edge = tuple(sorted((i, j)))
                    if edge in processed_edges:
                        continue
                    processed_edges.add(edge)

                    # Get both atom features
                    product_compound_idx = compound_pi + self.n_reactant_compounds
                    atom_pi_feat = self._get_enhanced_atom_features(
                        self.mol_prod, pi, product_compound_idx
                    )
                    atom_pj_feat = self._get_enhanced_atom_features(
                        self.mol_prod, pj, product_compound_idx
                    )
                    bond_feat = self.bond_featurizer(bond)
                    node_features = self._get_line_node_features(
                        atom_pi_feat, atom_pj_feat, bond_feat
                    )

                    # Create line node
                    self.line_nodes.append(edge)
                    self.line_node_features.append(node_features)
                    self.atom_origin_type.append(AtomOriginType.PRODUCT)
                    self.atom_compound_idx.append(product_compound_idx)
                    edge_map[edge] = current_idx
                    current_idx += 1

        # Step 2: Add self-loops for product atoms
        self._add_self_loops(is_reactant=False)

        # Step 3: Create line edges (analogous to reactants)
        processed_pairs = set()
        edges = list(edge_map.items())
        
        for (edge_a, idx1), (edge_b, idx2) in itertools.product(edges, edges):
            if idx1 == idx2:
                continue

            a_nodes = set(edge_a)
            b_nodes = set(edge_b)
            if a_nodes.intersection(b_nodes):
                pair = tuple(sorted((idx1, idx2)))
                if pair not in processed_pairs:
                    self.line_edges.extend([(idx1, idx2), (idx2, idx1)])
                    self.line_edge_features.extend([[0.0, 1.0], [0.0, 1.0]])
                    self.edge_origin_type.extend(
                        [EdgeOriginType.PRODUCT, EdgeOriginType.PRODUCT]
                    )
                    processed_pairs.add(pair)

    def _build_graph(self):
        """Build line graph representation."""
        self._build_reactant_line_graph()
        self._build_product_line_graph()

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

        if self.enthalpy is not None:
            data.enthalpy = torch.tensor([self.enthalpy], dtype=torch.float)

        # print(data.x.shape)
        # print(data.edge_index.shape)
        # print(data.edge_attr.shape)

        return data
