from typing import Dict, List, Optional

import hydra
import torch
import torch_geometric as tg
from omegaconf import DictConfig
from rdkit import Chem

from deeprxn.representation.rxn_graph import AtomOriginType, RxnGraphBase


class LineCGRGraph(RxnGraphBase):
    """Line graph representation of Condensed Graph of Reaction (CGR)."""

    def __init__(
        self,
        smiles: str,
        label: float,
        atom_featurizer: callable,
        bond_featurizer: callable,
        in_channel_multiplier: int = 2,
        pre_transform_cfg: Optional[DictConfig] = None,
        save_transform_features: bool = False,
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

        self.n_atoms = self.mol_reac.GetNumAtoms()
        self.pre_transform_cfg = pre_transform_cfg
        self.save_transform_features = save_transform_features
        self.component_features = {}
        self.merged_transform_features = {}

        self.line_nodes = []
        self.line_edges = []
        self.line_node_features = []
        self.line_edge_features = []
        self.atom_origin_type = []

        self.zero_bond_features = [0] * len(self.bond_featurizer(None))

        if self.pre_transform_cfg is not None:
            self._apply_component_transforms()
            self._merge_transform_features()

        self._build_graph()

    @staticmethod
    def _get_component_indices(origins) -> Dict[int, List[int]]:
        """Get atom indices for each component in a molecule."""
        component_indices = {}
        for atom_idx, origin in enumerate(origins):
            if origin not in component_indices:
                component_indices[origin] = []
            component_indices[origin].append(atom_idx)
        return component_indices

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
        """Apply transforms to each component."""
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
                for attr_name in attr_names:
                    if hasattr(data, attr_name):
                        if compound_idx not in self.component_features:
                            self.component_features[compound_idx] = {}
                        self.component_features[compound_idx][attr_name] = {
                            "features": getattr(data, attr_name),
                            "indices": indices,
                            "is_reactant": True,
                        }

            n_reactant_compounds = max(self.reac_origins) + 1
            for compound_idx, indices in self._get_component_indices(
                self.prod_origins
            ).items():
                data = self._create_component_graph(
                    self.mol_prod, indices, compound_idx
                )
                data = transform(data)
                for attr_name in attr_names:
                    if hasattr(data, attr_name):
                        if (
                            compound_idx + n_reactant_compounds
                            not in self.component_features
                        ):
                            self.component_features[
                                compound_idx + n_reactant_compounds
                            ] = {}
                        self.component_features[
                            compound_idx + n_reactant_compounds
                        ][attr_name] = {
                            "features": getattr(data, attr_name),
                            "indices": indices,
                            "is_reactant": False,
                        }

    def _merge_transform_features(self):
        """Merge transform features according to atom mapping while preserving CGR structure."""
        for compound_data in self.component_features.values():
            for attr_name in compound_data.keys():
                if attr_name not in self.merged_transform_features:
                    feat_shape = compound_data[attr_name]["features"].shape
                    if len(feat_shape) == 2:  # e.g., EigVecs (N, max_freqs)
                        merged_shape = (self.n_atoms, feat_shape[1] * 2)
                    elif (
                        len(feat_shape) == 3
                    ):  # e.g., EigVals (N, max_freqs, 1)
                        merged_shape = (
                            self.n_atoms,
                            feat_shape[1] * 2,
                            feat_shape[2],
                        )
                    else:  # Standard case
                        merged_shape = (self.n_atoms, feat_shape[1] * 2)

                    self.merged_transform_features[attr_name] = torch.zeros(
                        merged_shape, dtype=torch.float
                    )

        for attr_name in self.merged_transform_features.keys():
            for i in range(self.n_atoms):
                reac_feat = None
                prod_feat = None

                for compound_data in self.component_features.values():
                    if (
                        attr_name in compound_data
                        and compound_data[attr_name]["is_reactant"]
                    ):
                        if i in compound_data[attr_name]["indices"]:
                            local_idx = compound_data[attr_name][
                                "indices"
                            ].index(i)
                            reac_feat = compound_data[attr_name]["features"][
                                local_idx
                            ]

                prod_idx = self.ri2pi[i]
                for compound_data in self.component_features.values():
                    if (
                        attr_name in compound_data
                        and not compound_data[attr_name]["is_reactant"]
                    ):
                        if prod_idx in compound_data[attr_name]["indices"]:
                            local_idx = compound_data[attr_name][
                                "indices"
                            ].index(prod_idx)
                            prod_feat = compound_data[attr_name]["features"][
                                local_idx
                            ]

                if reac_feat is not None and prod_feat is not None:
                    # feat_diff = prod_feat - reac_feat
                    if len(reac_feat.shape) == 1:
                        self.merged_transform_features[attr_name][i] = (
                            torch.cat([reac_feat, prod_feat])
                        )
                    else:
                        self.merged_transform_features[attr_name][i] = (
                            torch.cat([reac_feat, prod_feat], dim=0)
                        )

    def _get_atom_features(self, atom_idx: int) -> List[float]:
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
    ) -> List[float]:
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

    def _build_line_graph(self):
        edge_map = {}
        current_idx = 0
        atoms_with_bonds = set()

        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                bond_reac = self.mol_reac.GetBondBetweenAtoms(i, j)
                bond_prod = self.mol_prod.GetBondBetweenAtoms(
                    self.ri2pi[i], self.ri2pi[j]
                )

                if bond_reac is not None or bond_prod is not None:
                    for src, tgt in [(i, j), (j, i)]:
                        source_atom_features = self._get_atom_features(src)
                        bond_features = self._get_bond_features(
                            bond_reac, bond_prod
                        )

                        if not self.save_transform_features:
                            transform_features = []
                            for (
                                attr_name,
                                features,
                            ) in self.merged_transform_features.items():
                                transform_features.extend(
                                    features[src].tolist()
                                )
                            node_features = (
                                source_atom_features
                                + bond_features
                                + transform_features
                            )
                        else:
                            node_features = (
                                source_atom_features + bond_features
                            )

                        self.line_nodes.append((src, tgt))
                        self.line_node_features.append(node_features)
                        self.atom_origin_type.append(
                            AtomOriginType.REACTANT_PRODUCT
                        )
                        edge_map[(src, tgt)] = current_idx
                        current_idx += 1
                        atoms_with_bonds.add(src)

        for i in range(self.n_atoms):
            if i not in atoms_with_bonds:
                atom_features = self._get_atom_features(i)

                transform_features = []
                for (
                    attr_name,
                    features,
                ) in self.merged_transform_features.items():
                    transform_features.extend(features[i].tolist())

                node_features = (
                    atom_features
                    + self.zero_bond_features
                    + transform_features
                )

                self.line_nodes.append((i, i))
                self.line_node_features.append(node_features)
                self.atom_origin_type.append(AtomOriginType.REACTANT_PRODUCT)
                edge_map[(i, i)] = current_idx
                current_idx += 1

        for (src1, tgt1), idx1 in edge_map.items():
            for (src2, tgt2), idx2 in edge_map.items():
                if tgt1 == src2 and idx1 != idx2:
                    self.line_edges.append((idx1, idx2))
                    self.line_edge_features.append([1.0])

    def _build_graph(self):
        """Build line CGR representation."""
        self._build_line_graph()

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

        if self.save_transform_features:
            for attr_name, features in self.merged_transform_features.items():
                setattr(data, attr_name, features)

        return data
