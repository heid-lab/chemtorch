from typing import Dict, List, Optional

import hydra
import torch
import torch_geometric as tg
from omegaconf import DictConfig
from rdkit import Chem

from deeprxn.representation.rxn_graph_base import (
    AtomOriginType,
    EdgeOriginType,
    RxnGraphBase,
)


class CGR(RxnGraphBase):
    """Condensed Graph of Reaction (CGR) representation."""

    def __init__(
        self,
        smiles: str,
        label: float,
        atom_featurizer: callable,
        bond_featurizer: callable,
        qm_featurizer= None,   
        single_featurizer= None,
        in_channel_multiplier: int = 2,  # TODO: look into this
        concat_transform_features: bool = False,
        pre_transform_cfg: Optional[DictConfig] = None,
        enthalpy=None,
    ):
        """Initialize CGR graph.

        Args:
            reaction_smiles: reaction string with atom mapping
            atom_featurizer: Function to generate atom features
            bond_featurizer: Function to generate bond features
        """
        super().__init__(
            smiles=smiles,
            label=label,
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
            enthalpy=enthalpy,
        )

        self.n_atoms = self.mol_reac.GetNumAtoms()
        self.pre_transform_cfg = pre_transform_cfg
        self.concat_transform_features = concat_transform_features
        self.component_features = {}
        self.merged_transform_features = {}

        if self.pre_transform_cfg is not None:
            self._apply_component_transforms()
            self._merge_transform_features()

        self._build_graph()

    def _create_component_graph(
        self, mol, atom_indices, compound_idx
    ) -> tg.data.Data:
        """Create a graph for a single component."""
        data = tg.data.Data()

        # Get features for atoms in this component
        x = []
        for idx in atom_indices:
            x.append(self.atom_featurizer(mol.GetAtomWithIdx(idx)))
        data.x = torch.tensor(x, dtype=torch.float)

        # Handle edges (only for components with multiple atoms)
        edges = []
        edge_features = []

        if len(atom_indices) > 1:
            # Create mapping from global to local indices
            global_to_local = {
                global_idx: local_idx
                for local_idx, global_idx in enumerate(atom_indices)
            }

            # Add bonds between atoms in this component
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

        # Store mapping information
        data.global_indices = torch.tensor(atom_indices, dtype=torch.long)
        data.compound_idx = compound_idx

        return data

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
                            "is_reactant": True,
                        }

            # Process product components
            n_reactant_compounds = max(self.reac_origins) + 1
            for compound_idx, indices in self._get_component_indices(
                self.prod_origins
            ).items():
                data = self._create_component_graph(
                    self.mol_prod, indices, compound_idx + n_reactant_compounds
                )
                data = transform(data)
                for attr_name in attr_names:
                    if hasattr(data, attr_name):
                        if (
                            compound_idx + n_reactant_compounds
                        ) not in self.component_features:
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

    @staticmethod
    def _get_component_indices(origins) -> Dict[int, List[int]]:
        """Get atom indices for each component in a molecule."""
        component_indices = {}
        for atom_idx, origin in enumerate(origins):
            if origin not in component_indices:
                component_indices[origin] = []
            component_indices[origin].append(atom_idx)
        return component_indices

    def _merge_transform_features(self):
        """Merge transform features according to atom mapping while preserving dimensions."""
        # Initialize merged features dictionary
        for compound_data in self.component_features.values():
            for attr_name in compound_data.keys():
                if attr_name not in self.merged_transform_features:
                    feat_shape = compound_data[attr_name]["features"].shape
                    # Initialize with correct dimensionality
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

        # Merge features
        for attr_name in self.merged_transform_features.keys():
            for i in range(self.n_atoms):
                # Find reactant feature
                reac_feat = None
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
                            break

                # Find corresponding product feature
                prod_idx = self.ri2pi[i]
                prod_feat = None
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
                            break

                if reac_feat is not None and prod_feat is not None:
                    # Ensure features have same shape
                    assert (
                        reac_feat.shape == prod_feat.shape
                    ), f"Feature shapes mismatch: {reac_feat.shape} vs {prod_feat.shape}"

                    # Handle different dimensionalities
                    if len(reac_feat.shape) == 1:  # 1D case
                        self.merged_transform_features[attr_name][i] = (
                            torch.cat([reac_feat, prod_feat])
                        )
                    elif (
                        len(reac_feat.shape) == 2
                    ):  # 2D case (e.g., EigVals with shape (max_freqs, 1))
                        # Concatenate along the first dimension
                        self.merged_transform_features[attr_name][i] = (
                            torch.cat([reac_feat, prod_feat], dim=0)
                        )
                    else:
                        raise ValueError(
                            f"Unsupported feature dimensionality: {len(reac_feat.shape)}"
                        )

    def _get_atom_features(self, atom_idx: int) -> List[float]:
        """Generate features for an atom in CGR representation.

        Concatenates reactant features with feature difference (prod - reac).

        Args:
            atom_idx: Index of atom in reactant molecule

        Returns:
            Combined feature vector [reac_feat, feat_diff]
        """
        f_atom_reac = self.atom_featurizer(
            self.mol_reac.GetAtomWithIdx(atom_idx)
        )
        f_atom_prod = self.atom_featurizer(
            self.mol_prod.GetAtomWithIdx(self.ri2pi[atom_idx])
        )
        f_atom_diff = [y - x for x, y in zip(f_atom_reac, f_atom_prod)]
        return f_atom_reac + f_atom_diff

    def _get_bond_features(
        self, bond_reac: Chem.Bond, bond_prod: Chem.Bond
    ) -> List[float]:
        """Generate features for a bond in CGR representation.

        Concatenates reactant features with feature difference (prod - reac).

        Args:
            bond_reac: Bond in reactant molecule (or None)
            bond_prod: Bond in product molecule (or None)

        Returns:
            Combined feature vector [reac_feat, feat_diff]
        """
        f_bond_reac = (
            self.bond_featurizer(bond_reac)
            if bond_reac
            else [0] * len(self.bond_featurizer(None))
        )
        f_bond_prod = (
            self.bond_featurizer(bond_prod)
            if bond_prod
            else [0] * len(self.bond_featurizer(None))
        )
        f_bond_diff = [y - x for x, y in zip(f_bond_reac, f_bond_prod)]
        return (
            f_bond_reac + f_bond_diff
        )  # concatenate the reactant and product bond features

    def _build_graph(self):
        """Build CGR representation.

        Creates a single graph where:
        - Each atom appears once
        - Atom features combine reactant state and change
        - Bond features combine reactant state and change
        """
        for i in range(self.n_atoms):
            self.f_atoms.append(self._get_atom_features(i))
            self.atom_origin_type.append(AtomOriginType.REACTANT_PRODUCT)

            for j in range(i + 1, self.n_atoms):
                bond_reac = self.mol_reac.GetBondBetweenAtoms(i, j)
                bond_prod = self.mol_prod.GetBondBetweenAtoms(
                    self.ri2pi[i], self.ri2pi[j]
                )

                # only add if bond exists in either reactants or products
                if bond_reac is None and bond_prod is None:
                    continue

                f_bond = self._get_bond_features(bond_reac, bond_prod)
                # add bond in both directions (i->j and j->i)
                self.f_bonds.extend([f_bond, f_bond])
                self.edge_index.extend([(i, j), (j, i)])
                self.edge_origin_type.extend(
                    [
                        EdgeOriginType.REACTANT_PRODUCT,
                        EdgeOriginType.REACTANT_PRODUCT,
                    ]
                )

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

        if self.enthalpy is not None:
            data.enthalpy = torch.tensor([self.enthalpy], dtype=torch.float)

        if self.concat_transform_features:
            for features in self.merged_transform_features.values():
                data.x = torch.cat([data.x, features], dim=1)
        else:
            for attr_name, features in self.merged_transform_features.items():
                setattr(data, attr_name, features)

        return data
