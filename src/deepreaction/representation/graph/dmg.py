from typing import Dict, List, Optional, Tuple

import torch

from rdkit.Chem import Atom, Bond
from torch_geometric.data import Data

from deepreaction.featurizer.featurizer_base import FeaturizerBase
from deepreaction.featurizer.featurizer_compose import FeaturizerCompose
from deepreaction.representation.abstract_representation import AbstractRepresentation
from deepreaction.representation.graph.graph_reprs_utils import (
    AtomOriginType,
    EdgeOriginType,
    make_mol,
    map_reac_to_prod,
)

class DMG(AbstractRepresentation[Data]):
    """
    Stateless class for constructing Dual Molecular Graph (DMG) representations.

    This class does not hold any data itself. Instead, it provides a `forward()` method
    that takes a sample (e.g., a dict or pd.Series) and returns a PyTorch Geometric Data object
    representing the reaction as a dual molecular graph.

    Usage:
        >>> dmg = DMG(atom_featurizer, bond_featurizer, ...)
        >>> data = dmg.construct(sample)
        >>> data = dmg(sample)
    """

    def __init__(
        self,
        # TODO: Rename atom_featurizer and bond_featurizer to node_featurizer and edge_featurizer
        atom_featurizer: FeaturizerBase[Atom] | FeaturizerCompose,
        bond_featurizer: FeaturizerBase[Bond] | FeaturizerCompose,
        # TODO: Make this an optional dict of property: featurizer
        qm_featurizer: Optional[FeaturizerBase[Atom] | FeaturizerCompose] = None,
        single_featurizer: Optional[FeaturizerBase[Atom] | FeaturizerCompose] = None,
        connection_direction: str = "bidirectional",
        concat_origin_feature: bool = False,
        pre_transform_list: Optional[List] = None,  # TODO: Remove
        extra_zero_fvec: bool = False,
        **kwargs # ignored, TODO: remove once all featurizers are passed explicitly
    ):
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.qm_featurizer = qm_featurizer
        self.single_featurizer = single_featurizer

        self.qm_f = []
        self.connection_direction = connection_direction
        self.concat_origin_feature = concat_origin_feature
        self.pre_transform_list = pre_transform_list or []
        self.extra_zero_fvec = extra_zero_fvec

    # TODO: Break this function into smaller methods and reuse them in MolGraph class
    @override
    def construct(self, smiles: str) -> Data:
        # Parse reactant and product SMILES
        smiles_reac, _, smiles_prod = smiles.split(">")

        mol_reac, reac_origins = make_mol(smiles_reac)
        mol_prod, prod_origins = make_mol(smiles_prod)
        ri2pi = map_reac_to_prod(mol_reac, mol_prod)

        n_atoms_reac = mol_reac.GetNumAtoms()
        n_atoms_prod = mol_prod.GetNumAtoms()
        n_atoms = n_atoms_reac + n_atoms_prod

        # Track number of unique reactant molecules for offset
        n_reactant_compounds = max(reac_origins) + 1 if reac_origins else 1
        atom_compound_idx: List[int] = []

        # Track the mapping between original indices and final graph indices
        original_to_graph_indices: Dict[Tuple[int, int], int] = {}

        # Atom features and types
        f_atoms: List[List[float]] = []
        atom_origin_type: List[AtomOriginType] = []
        qm_f: List[List[float]] = []
        single_f: List[List[float]] = []

        # Edge features and types
        edge_index: List[Tuple[int, int]] = []
        f_bonds: List[List[float]] = []
        edge_origin_type: List[EdgeOriginType] = []

        # --- Build reactant graph ---
        # TODO: Delegate to MolGraph class 
        current_idx = 0
        for i in range(n_atoms_reac):
            f_atoms.append(self.atom_featurizer(mol_reac.GetAtomWithIdx(i)))
            if self.qm_featurizer is not None:
                qm_f.append(self.qm_featurizer(mol_reac.GetAtomWithIdx(i)))
            if self.single_featurizer is not None:
                single_f.append(self.single_featurizer(mol_reac.GetAtomWithIdx(i)))
            atom_origin_type.append(AtomOriginType.REACTANT)
            compound_idx = reac_origins[i] if reac_origins else 0
            atom_compound_idx.append(compound_idx)
            original_to_graph_indices[(compound_idx, i)] = current_idx
            current_idx += 1

            for j in range(i + 1, n_atoms_reac):
                bond = mol_reac.GetBondBetweenAtoms(i, j)
                if bond:
                    f_bond = self.bond_featurizer(bond)
                    f_bonds.extend([f_bond, f_bond])
                    edge_index.extend([(i, j), (j, i)])
                    edge_origin_type.extend([EdgeOriginType.REACTANT, EdgeOriginType.REACTANT])

        # --- Build product graph ---
        # TODO: Delegate to MolGraph class 
        offset = n_atoms_reac
        current_idx = offset
        for i in range(n_atoms_prod):
            prod_idx = ri2pi[i]
            f_atoms.append(self.atom_featurizer(mol_prod.GetAtomWithIdx(prod_idx)))
            if self.qm_featurizer is not None:
                qm_f.append(self.qm_featurizer(mol_prod.GetAtomWithIdx(prod_idx)))
            if self.single_featurizer is not None:
                single_f.append(self.single_featurizer(mol_prod.GetAtomWithIdx(prod_idx)))
            atom_origin_type.append(AtomOriginType.PRODUCT)
            compound_idx = (prod_origins[prod_idx] + n_reactant_compounds) if prod_origins else 0
            atom_compound_idx.append(compound_idx)
            original_to_graph_indices[(compound_idx, prod_idx)] = current_idx
            current_idx += 1

            for j in range(i + 1, n_atoms_prod):
                bond = mol_prod.GetBondBetweenAtoms(ri2pi[i], ri2pi[j])
                if bond:
                    f_bond = self.bond_featurizer(bond)
                    f_bonds.extend([f_bond, f_bond])
                    edge_index.extend(
                        [(i + offset, j + offset), (j + offset, i + offset)]
                    )
                    edge_origin_type.extend([EdgeOriginType.PRODUCT, EdgeOriginType.PRODUCT])

        # --- Connect graphs ---
        if self.connection_direction is not None:
            f_bond = self.bond_featurizer(None)
            for i in range(n_atoms_reac):
                if self.connection_direction == "bidirectional":
                    f_bonds.extend([f_bond, f_bond])
                    edge_index.extend([(i, i + offset), (i + offset, i)])
                    edge_origin_type.extend(
                        [EdgeOriginType.REACTANT_PRODUCT, EdgeOriginType.REACTANT_PRODUCT]
                    )
                elif self.connection_direction == "reactants_to_products":
                    f_bonds.append(f_bond)
                    edge_index.append((i, i + offset))
                    edge_origin_type.append(EdgeOriginType.REACTANT_PRODUCT)
                elif self.connection_direction == "products_to_reactants":
                    f_bonds.append(f_bond)
                    edge_index.append((i + offset, i))
                    edge_origin_type.append(EdgeOriginType.REACTANT_PRODUCT)

        # --- Build PyG Data object ---
        data = Data()
        data.x = torch.tensor(f_atoms, dtype=torch.float)
        data.edge_index = (
            torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            if edge_index else torch.zeros((2, 0), dtype=torch.long)
        )
        data.edge_attr = (
            torch.tensor(f_bonds, dtype=torch.float)
            if f_bonds else torch.zeros((0, len(self.bond_featurizer(None))), dtype=torch.float)
        )
        data.smiles = smiles
        data.atom_origin_type = torch.tensor(atom_origin_type, dtype=torch.long)
        data.atom_compound_idx = torch.tensor(atom_compound_idx, dtype=torch.long)
        data.edge_origin_type = torch.tensor(edge_origin_type, dtype=torch.long)
        data.num_nodes = n_atoms

        if self.extra_zero_fvec:
            n, d = data.x.shape
            new_x = torch.zeros(n, 2 * d)
            reactant_mask = (data.atom_origin_type == AtomOriginType.REACTANT.value)
            product_mask = (data.atom_origin_type == AtomOriginType.PRODUCT.value)
            new_x[reactant_mask, d:] = data.x[reactant_mask]
            new_x[product_mask, :d] = data.x[product_mask]
            data.x = new_x

        if self.qm_featurizer is not None and qm_f:
            data.qm_f = torch.tensor(qm_f, dtype=torch.float)
        if self.single_featurizer is not None and single_f:
            data.single_f = torch.tensor(single_f, dtype=torch.float)

        # Optionally, add origin encodings
        node_encodings = torch.tensor(
            [self._get_node_type_encoding(t) for t in atom_origin_type],
            dtype=torch.float,
        )
        edge_encodings = torch.tensor(
            [self._get_edge_type_encoding(t) for t in edge_origin_type],
            dtype=torch.float,
        )
        if self.concat_origin_feature:
            data.x = torch.cat([data.x, node_encodings], dim=1)
            data.edge_attr = torch.cat([data.edge_attr, edge_encodings], dim=1)
        else:
            data.node_origin_encoding = node_encodings
            data.edge_origin_encoding = edge_encodings

        # Optionally, apply pre-transforms to components
        if self.pre_transform_list:
            component_features = {}
            for transform in self.pre_transform_list:
                attr_names = transform.attr_name
                if isinstance(attr_names, str):
                    attr_names = [attr_names]
                # Reactant components
                for compound_idx, indices in self._get_component_indices(reac_origins).items():
                    comp_data = self._create_component_graph(mol_reac, indices, compound_idx)
                    comp_data = transform(comp_data)
                    for attr_name in attr_names:
                        if hasattr(comp_data, attr_name):
                            if compound_idx not in component_features:
                                component_features[compound_idx] = {}
                            component_features[compound_idx][attr_name] = {
                                "features": getattr(comp_data, attr_name),
                                "indices": indices,
                                "offset": 0,
                            }
                # Product components
                for compound_idx, indices in self._get_component_indices(prod_origins).items():
                    comp_data = self._create_component_graph(
                        mol_prod, indices, compound_idx + n_reactant_compounds
                    )
                    comp_data = transform(comp_data)
                    for attr_name in attr_names:
                        if hasattr(comp_data, attr_name):
                            compound_key = compound_idx + n_reactant_compounds
                            if compound_key not in component_features:
                                component_features[compound_key] = {}
                            component_features[compound_key][attr_name] = {
                                "features": getattr(comp_data, attr_name),
                                "indices": indices,
                                "offset": n_atoms_reac,
                            }
            # Assign features to data
            for attr_name in set(
                attr
                for comp_features in component_features.values()
                for attr in comp_features.keys()
            ):
                sample_features = next(
                    feat["features"]
                    for comp_features in component_features.values()
                    for name, feat in comp_features.items()
                    if name == attr_name
                )
                feature_shape = list(sample_features.shape[1:])
                combined_shape = [n_atoms] + feature_shape
                combined_features = torch.zeros(combined_shape, dtype=torch.float)
                for comp_idx, comp_features in component_features.items():
                    if attr_name in comp_features:
                        feat_data = comp_features[attr_name]
                        for local_idx, orig_idx in enumerate(feat_data["indices"]):
                            graph_idx = original_to_graph_indices.get((comp_idx, orig_idx), None)
                            if graph_idx is not None:
                                if len(feature_shape) == 1:
                                    combined_features[graph_idx] = feat_data["features"][local_idx]
                                else:
                                    combined_features[graph_idx, :] = feat_data["features"][local_idx, :]
                setattr(data, attr_name, combined_features)

        return data

    @staticmethod
    def _get_node_type_encoding(origin_type: AtomOriginType) -> List[float]:
        encoding = [0.0] * len(AtomOriginType)
        encoding[origin_type.value] = 1.0
        return encoding

    @staticmethod
    def _get_edge_type_encoding(origin_type: EdgeOriginType) -> List[float]:
        encoding = [0.0] * len(EdgeOriginType)
        encoding[origin_type.value] = 1.0
        return encoding

    @staticmethod
    def _get_component_indices(origins) -> Dict[int, List[int]]:
        component_indices = {}
        for atom_idx, origin in enumerate(origins):
            if origin not in component_indices:
                component_indices[origin] = []
            component_indices[origin].append(atom_idx)
        return component_indices

    def _create_component_graph(self, mol, atom_indices, compound_idx) -> Data:
        data = Data()
        x = [self.atom_featurizer(mol.GetAtomWithIdx(idx)) for idx in atom_indices]
        data.x = torch.tensor(x, dtype=torch.float)
        edges = []
        edge_features = []
        if len(atom_indices) > 1:
            global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(atom_indices)}
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
            if edges else torch.zeros((2, 0), dtype=torch.long)
        )
        data.edge_attr = (
            torch.tensor(edge_features, dtype=torch.float)
            if edge_features else torch.zeros((0, len(self.bond_featurizer(None))), dtype=torch.float)
        )
        data.num_nodes = len(atom_indices)
        data.global_indices = torch.tensor(atom_indices, dtype=torch.long)
        data.compound_idx = compound_idx
        return data
