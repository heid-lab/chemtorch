from typing import Dict, List, Optional, Tuple

import hydra
import torch
import torch_geometric as tg
from omegaconf import DictConfig

from deeprxn.representation.rxn_graph_base import (
    AtomOriginType,
    EdgeOriginType,
    RxnGraphBase,
)


class DMG(RxnGraphBase):
    """Dual Molecular Graph (DMG) representation."""

    def __init__(
        self,
        smiles: str,
        label: float,
        atom_featurizer: callable,
        bond_featurizer: callable,
        qm_featurizer: callable,
        connection_direction: str = "bidirectional",
        concat_origin_feature: bool = False,
        in_channel_multiplier: int = 1,
        pre_transform_cfg: Optional[Dict[str, DictConfig]] = None,
        enthalpy=None,
    ):
        """Initialize graph.

        Args:
            reaction_smiles: SMARTS reaction string with atom mapping
            atom_featurizer: Function to generate atom features
            bond_featurizer: Function to generate bond features
            connection_direction: How to connect corresponding atoms:
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
            enthalpy=enthalpy,
        )
        self.connection_direction = connection_direction
        self.concat_origin_feature = concat_origin_feature
        self.pre_transform_cfg = pre_transform_cfg
        self.component_features = {}

        self.qm_featurizer = qm_featurizer
        self.qm_f = []

        self.n_atoms_reac = self.mol_reac.GetNumAtoms()
        self.n_atoms_prod = self.mol_prod.GetNumAtoms()
        self.n_atoms = self.n_atoms_reac + self.n_atoms_prod

        # track number of unique reactant molecules for offset
        self.n_reactant_compounds = max(self.reac_origins) + 1
        self.atom_compound_idx: List[int] = []  # TODO: make optional?

        # Track the mapping between original indices and final graph indices
        self.original_to_graph_indices: Dict[Tuple[int, int], int] = (
            {}
        )  # (compound_idx, local_idx) -> graph_idx

        if self.pre_transform_cfg is not None:
            self._apply_component_transforms()

        # build connected pair graph
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

    @staticmethod
    def _get_node_type_encoding(origin_type: AtomOriginType) -> List[float]:
        """Convert atom origin type to one-hot encoding."""
        encoding = [0.0] * len(AtomOriginType)
        encoding[origin_type.value] = 1.0
        return encoding

    @staticmethod
    def _get_edge_type_encoding(origin_type: EdgeOriginType) -> List[float]:
        """Convert edge origin type to one-hot encoding."""
        encoding = [0.0] * len(EdgeOriginType)
        encoding[origin_type.value] = 1.0
        return encoding

    @staticmethod
    def _get_component_indices(origins) -> Dict[int, List[int]]:
        """Get atom indices for each component in a molecule."""
        component_indices = {}
        for atom_idx, origin in enumerate(origins):
            if origin not in component_indices:
                component_indices[origin] = []
            component_indices[origin].append(atom_idx)
        return component_indices

    def _update_index_mapping(
        self, compound_idx: int, local_idx: int, graph_idx: int
    ):
        """Update the mapping between original and graph indices."""
        self.original_to_graph_indices[(compound_idx, local_idx)] = graph_idx

    def _apply_component_transforms(self):
        """Apply transforms to each component separately."""
        for _, config in self.pre_transform_cfg.items():
            transform = hydra.utils.instantiate(config)
            attr_names = transform.attr_name
            if isinstance(attr_names, str):
                attr_names = [attr_names]

            # Process reactant components
            for compound_idx, indices in self._get_component_indices(
                self.reac_origins
            ).items():
                data = self._create_component_graph(
                    self.mol_reac, indices, compound_idx
                )
                data = transform(data)
                for attr_name in attr_names:
                    if hasattr(data, attr_name):
                        if compound_idx not in self.component_features:
                            self.component_features[compound_idx] = {}
                        self.component_features[compound_idx][attr_name] = {
                            "features": getattr(data, attr_name),
                            "indices": indices,
                            "offset": 0,
                        }

            # Process product components
            for compound_idx, indices in self._get_component_indices(
                self.prod_origins
            ).items():
                data = self._create_component_graph(
                    self.mol_prod,
                    indices,
                    compound_idx + self.n_reactant_compounds,
                )
                data = transform(data)
                for attr_name in attr_names:
                    if hasattr(data, attr_name):
                        compound_key = compound_idx + self.n_reactant_compounds
                        if compound_key not in self.component_features:
                            self.component_features[compound_key] = {}
                        self.component_features[compound_key][attr_name] = {
                            "features": getattr(data, attr_name),
                            "indices": indices,
                            "offset": self.n_atoms_reac,
                        }

    def _build_reactant_graph(self):
        """Build graph for reactant molecules."""
        current_idx = 0

        # Add reactant atom features
        for i in range(self.n_atoms_reac):
            self.f_atoms.append(
                self.atom_featurizer(self.mol_reac.GetAtomWithIdx(i))
            )
            if self.qm_featurizer is not None:
                self.qm_f.append(self.qm_featurizer(self.mol_reac.GetAtomWithIdx(i)))
            self.atom_origin_type.append(AtomOriginType.REACTANT)
            compound_idx = self.reac_origins[i]
            self.atom_compound_idx.append(compound_idx)

            self._update_index_mapping(compound_idx, i, current_idx)
            current_idx += 1

            # Add reactant bonds
            for j in range(i + 1, self.n_atoms_reac):
                bond = self.mol_reac.GetBondBetweenAtoms(i, j)
                if bond:
                    f_bond = self.bond_featurizer(bond)
                    self.f_bonds.extend([f_bond, f_bond])
                    self.edge_index.extend([(i, j), (j, i)])
                    self.edge_origin_type.extend(
                        [EdgeOriginType.REACTANT, EdgeOriginType.REACTANT]
                    )

    def _build_product_graph(self):
        """Build graph for product molecules."""
        offset = self.n_atoms_reac  # Offset for product atom indices
        current_idx = offset

        # Add product atom features
        for i in range(self.n_atoms_prod):
            prod_idx = self.ri2pi[i]
            self.f_atoms.append(
                self.atom_featurizer(self.mol_prod.GetAtomWithIdx(prod_idx))
            )
            if self.qm_featurizer is not None:
                self.qm_f.append(
                    self.qm_featurizer(self.mol_prod.GetAtomWithIdx(prod_idx))
                )
            self.atom_origin_type.append(AtomOriginType.PRODUCT)
            compound_idx = (
                self.prod_origins[prod_idx] + self.n_reactant_compounds
            )
            self.atom_compound_idx.append(compound_idx)
            self._update_index_mapping(compound_idx, prod_idx, current_idx)
            current_idx += 1

            # Add product bonds
            for j in range(i + 1, self.n_atoms_prod):
                bond = self.mol_prod.GetBondBetweenAtoms(
                    self.ri2pi[i], self.ri2pi[j]
                )
                if bond:
                    f_bond = self.bond_featurizer(bond)
                    self.f_bonds.extend([f_bond, f_bond])
                    self.edge_index.extend(
                        [(i + offset, j + offset), (j + offset, i + offset)]
                    )
                    self.edge_origin_type.extend(
                        [EdgeOriginType.PRODUCT, EdgeOriginType.PRODUCT]
                    )

    def _connect_graphs(self):
        """Add edges connecting corresponding atoms in reactants and products."""
        if self.connection_direction == None:
            return

        offset = self.n_atoms_reac

        # Zero vector for connecting edge features
        f_bond = [0] * len(self.bond_featurizer(None))

        for i in range(self.n_atoms_reac):
            if self.connection_direction == "bidirectional":
                self.f_bonds.extend([f_bond, f_bond])
                self.edge_index.extend([(i, i + offset), (i + offset, i)])
                self.edge_origin_type.extend(
                    [
                        EdgeOriginType.REACTANT_PRODUCT,
                        EdgeOriginType.REACTANT_PRODUCT,
                    ]
                )
            elif self.connection_direction == "reactants_to_products":
                self.f_bonds.append(f_bond)
                self.edge_index.append((i, i + offset))
                self.edge_origin_type.append(EdgeOriginType.REACTANT_PRODUCT)
            elif self.connection_direction == "products_to_reactants":
                self.f_bonds.append(f_bond)
                self.edge_index.append((i + offset, i))
                self.edge_origin_type.append(EdgeOriginType.REACTANT_PRODUCT)

    def _build_graph(self):
        """Build connected pair representation.

        Creates two separate graphs for reactants and products,
        then optionally connects corresponding atoms.
        """
        self._build_reactant_graph()
        self._build_product_graph()
        self._connect_graphs()

    def to_pyg_data(self) -> tg.data.Data:
        """Convert the molecular graph to a PyTorch Geometric Data object."""
        data = tg.data.Data()
        data.x = torch.tensor(self.f_atoms, dtype=torch.float)
        data.edge_index = (
            torch.tensor(self.edge_index, dtype=torch.long).t().contiguous()
        )
        data.edge_attr = torch.tensor(self.f_bonds, dtype=torch.float)
        data.y = torch.tensor([self.label], dtype=torch.float)
        data.smiles = self.smiles
        data.atom_origin_type = torch.tensor(
            self.atom_origin_type, dtype=torch.long
        )
        data.atom_compound_idx = torch.tensor(
            self.atom_compound_idx, dtype=torch.long
        )
        data.edge_origin_type = torch.tensor(
            self.edge_origin_type, dtype=torch.long
        )

        if self.qm_featurizer is not None:
            data.qm_f = torch.tensor(self.qm_f, dtype=torch.float)

        node_encodings = torch.tensor(
            [self._get_node_type_encoding(t) for t in self.atom_origin_type],
            dtype=torch.float,
        )

        edge_encodings = torch.tensor(
            [self._get_edge_type_encoding(t) for t in self.edge_origin_type],
            dtype=torch.float,
        )

        if self.enthalpy is not None:
            data.enthalpy = torch.tensor([self.enthalpy], dtype=torch.float)

        if self.concat_origin_feature == True:
            # Concatenate with existing features
            data.x = torch.cat([data.x, node_encodings], dim=1)
            data.edge_attr = torch.cat([data.edge_attr, edge_encodings], dim=1)
        else:
            # Store separately
            data.node_origin_encoding = node_encodings
            data.edge_origin_encoding = edge_encodings

        # Handle pre-transform attributes using the index mapping
        for attr_name in set(
            attr
            for comp_features in self.component_features.values()
            for attr in comp_features.keys()
        ):
            # Get the shape from the first non-empty feature tensor
            sample_features = next(
                feat["features"]
                for comp_features in self.component_features.values()
                for name, feat in comp_features.items()
                if name == attr_name
            )

            # Initialize combined_features with the correct shape
            feature_shape = list(
                sample_features.shape[1:]
            )  # Get all dimensions except the first
            combined_shape = [self.n_atoms] + feature_shape
            combined_features = torch.zeros(combined_shape, dtype=torch.float)

            for comp_idx, comp_features in self.component_features.items():
                if attr_name in comp_features:
                    feat_data = comp_features[attr_name]
                    for local_idx, orig_idx in enumerate(feat_data["indices"]):
                        graph_idx = self.original_to_graph_indices.get(
                            (comp_idx, orig_idx), None
                        )
                        if graph_idx is not None:
                            if len(feature_shape) == 1:
                                combined_features[graph_idx] = feat_data[
                                    "features"
                                ][local_idx]
                            else:
                                combined_features[graph_idx, :] = feat_data[
                                    "features"
                                ][local_idx, :]

            setattr(data, attr_name, combined_features)

        return data
