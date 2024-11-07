from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch

from deeprxn.representation.rxn_graph import AtomOriginType
from deeprxn.transform.transform import TransformBase


class DummyTransform(TransformBase):
    """Transform for adding dummy nodes to reaction graphs."""

    def __init__(
        self,
        dummy_node: Optional[str] = None,
        dummy_connection: str = "to_dummy",
        dummy_dummy_connection: str = "bidirectional",
        dummy_feat_init: str = "zeros",
    ):
        """Initialize dummy transform.

        Args:
            dummy_node: Type of dummy node(s) to add. One of [None, "global",
                "reactant_product", "all_separate"]
            dummy_connection: How to connect dummy nodes. One of ["to_dummy",
                "from_dummy", "bidirectional"]
            dummy_dummy_connection: How to connect dummy nodes to each other
            dummy_feat_init: How to initialize dummy features ["zeros", "ones"]
            enabled: Whether transform is enabled
        """
        super().__init__()

        valid_dummy_nodes = [
            None,
            "global",
            "reactant_product",
            "all_separate",
        ]
        if dummy_node not in valid_dummy_nodes:
            raise ValueError(
                f"Invalid dummy_node. Choose from: {', '.join(map(str, valid_dummy_nodes))}"
            )

        valid_dummy_connections = ["to_dummy", "from_dummy", "bidirectional"]
        if dummy_connection not in valid_dummy_connections:
            raise ValueError(
                f"Invalid dummy_connection. Choose from: {', '.join(valid_dummy_connections)}"
            )

        self.dummy_node = dummy_node
        self.dummy_connection = dummy_connection
        self.dummy_dummy_connection = dummy_dummy_connection
        self.dummy_feat_init = dummy_feat_init

        print(f"DummyTransform: {self.dummy_node}")

    def __call__(self, data: Any) -> Any:
        """Apply dummy node transform to graph data.

        Args:
            data: Graph data object containing:
                - f_atoms: List of atom features
                - f_bonds: List of bond features
                - edge_index: List of edge indices
                - atom_origins: List of atom origins
                - atom_origin_type: List of atom origin types
                - is_real_bond: List of boolean flags for real bonds
                - representation: Graph representation type
                - n_atoms: Number of atoms

        Returns:
            Modified graph data with dummy nodes added
        """
        if not self.enabled or not self.dummy_node:
            return data

        # Get feature dimensions
        len_node_feat = len(data.f_atoms[0])
        len_bond_feat = len(data.f_bonds[0]) if data.f_bonds else 0

        # Adjust feature lengths for CGR representation
        if data.representation == "CGR":
            if self.dummy_node != "global":
                raise ValueError(
                    "CGR representation only supports global dummy node"
                )
            len_node_feat *= 2
            len_bond_feat *= 2

        # Initialize dummy features
        dummy_feature = (
            torch.zeros(len_node_feat)
            if self.dummy_feat_init == "zeros"
            else torch.ones(len_node_feat)
        )
        f_bond = [0] * len_bond_feat

        # Add appropriate dummy nodes
        if self.dummy_node == "global":
            self._add_global_dummy(data, dummy_feature, f_bond)
        elif self.dummy_node == "reactant_product":
            self._add_reactant_product_dummies(data, dummy_feature, f_bond)
        elif self.dummy_node == "all_separate":
            self._add_all_separate_dummies(data, dummy_feature, f_bond)

        return data

    def _connect_dummy_to_node(
        self, data: Any, dummy_idx: int, node_idx: int, f_bond: List[float]
    ) -> None:
        """Connect a dummy node to a regular node."""
        if self.dummy_connection in [
            "bidirectional",
            "from_dummy",
        ]:
            data.f_bonds.append(f_bond)
            data.edge_index.append((dummy_idx, node_idx))
            data.is_real_bond.append(False)
        if self.dummy_connection in [
            "bidirectional",
            "to_dummy",
        ]:
            data.f_bonds.append(f_bond)
            data.edge_index.append((node_idx, dummy_idx))
            data.is_real_bond.append(False)

    def _add_global_dummy(
        self, data: Any, dummy_feature: torch.Tensor, f_bond: List[float]
    ) -> None:
        """Add a single global dummy node."""
        dummy_idx = len(data.f_atoms)
        data.f_atoms.append(dummy_feature.tolist())
        data.atom_origins.append(-1)
        data.atom_origin_type.append(AtomOriginType.DUMMY)

        dummy_connections = data.n_atoms
        if data.representation == "connected_pair":
            dummy_connections *= 2

        for i in range(dummy_connections):
            self._connect_dummy_to_node(data, dummy_idx, i, f_bond)

    def _add_reactant_product_dummies(
        self, data: Any, dummy_feature: torch.Tensor, f_bond: List[float]
    ) -> None:
        """Add separate dummy nodes for reactants and products."""
        # Add reactant dummy
        dummy_reactant_idx = len(data.f_atoms)
        data.f_atoms.append(dummy_feature.tolist())
        data.atom_origins.append(-1)
        data.atom_origin_type.append(AtomOriginType.DUMMY)

        # Add product dummy
        dummy_product_idx = len(data.f_atoms)
        data.f_atoms.append(dummy_feature.tolist())
        data.atom_origins.append(-1)
        data.atom_origin_type.append(AtomOriginType.DUMMY)

        # Connect to atoms
        for i in range(data.n_atoms):
            self._connect_dummy_to_node(data, dummy_reactant_idx, i, f_bond)
            self._connect_dummy_to_node(
                data, dummy_product_idx, i + data.n_atoms, f_bond
            )

        # Connect dummies if specified
        if self.dummy_dummy_connection == "bidirectional":
            data.f_bonds.extend([f_bond, f_bond])
            data.edge_index.extend(
                [
                    (dummy_reactant_idx, dummy_product_idx),
                    (dummy_product_idx, dummy_reactant_idx),
                ]
            )
            data.is_real_bond.extend([False, False])

    def _add_all_separate_dummies(
        self, data: Any, dummy_feature: torch.Tensor, f_bond: List[float]
    ) -> None:
        """Add separate dummy nodes for each molecule."""
        unique_origins = set(data.atom_origins)
        dummy_indices = {}

        # Create dummy nodes
        for origin in unique_origins:
            dummy_idx = len(data.f_atoms)
            data.f_atoms.append(dummy_feature.tolist())
            data.atom_origins.append(-1)
            data.atom_origin_type.append(AtomOriginType.DUMMY)
            dummy_indices[origin] = dummy_idx

        # Connect atoms to their respective dummies
        for i, origin in enumerate(data.atom_origins):
            if origin in dummy_indices:
                self._connect_dummy_to_node(
                    data, dummy_indices[origin], i, f_bond
                )

        # Connect dummies if specified
        if self.dummy_dummy_connection == "bidirectional":
            dummy_list = list(dummy_indices.values())
            for i in range(len(dummy_list)):
                for j in range(i + 1, len(dummy_list)):
                    data.f_bonds.extend([f_bond, f_bond])
                    data.edge_index.extend(
                        [
                            (dummy_list[i], dummy_list[j]),
                            (dummy_list[j], dummy_list[i]),
                        ]
                    )
                    data.is_real_bond.extend([False, False])
