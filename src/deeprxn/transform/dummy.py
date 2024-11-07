from enum import Enum
from typing import List, Optional

import torch
from omegaconf import DictConfig

from deeprxn.representation.cgr_graph import CGRGraph
from deeprxn.representation.connected_pair_graph import ConnectedPairGraph
from deeprxn.representation.rxn_graph import AtomOriginType, RxnGraphBase
from deeprxn.transform.transform import TransformBase


class DummyNodeTransform(TransformBase):
    """Transform for adding dummy nodes to reaction graphs."""

    def __init__(
        self,
        mode: str,
        connection_type: Optional[str] = None,
        dummy_dummy_connection: Optional[str] = None,
        feature_init: str = "zeros",
    ):
        super().__init__()
        self.mode = mode
        self.connection_type = connection_type
        self.dummy_dummy_connection = dummy_dummy_connection
        self.feature_init = feature_init

    def __call__(self, graph) -> None:
        """Apply dummy node transform to the graph."""
        if isinstance(graph, CGRGraph) and self.mode != "global":
            raise ValueError("CGR graphs only support global dummy node mode")

        # feature dimensions from existing atoms/bonds
        node_feat_dim = len(graph.f_atoms[0])
        bond_feat_dim = len(graph.f_bonds[0]) if graph.f_bonds else 0

        # dummy feature vector
        dummy_feature = (
            torch.zeros(node_feat_dim)
            if self.feature_init == "zeros"
            else torch.ones(node_feat_dim)
        )

        # dummy bond feature vector
        dummy_bond = [0] * bond_feat_dim

        if self.mode == "global":
            self._add_global_dummy(graph, dummy_feature, dummy_bond)
        elif self.mode == "reactant_product":
            self._add_reactant_product_dummies(
                graph, dummy_feature, dummy_bond
            )
        else:
            raise ValueError(f"Unknown dummy node mode: {self.mode}")

    def _connect_dummy_to_node(
        self,
        graph,
        dummy_idx: int,
        node_idx: int,
        bond_feat: List[float],
    ) -> None:
        """Helper method to connect dummy node to a regular node."""
        if self.connection_type in [
            "bidirectional",
            "from_dummy",
        ]:
            graph.f_bonds.append(bond_feat)
            graph.edge_index.append((dummy_idx, node_idx))

        if self.connection_type in [
            "bidirectional",
            "to_dummy",
        ]:
            graph.f_bonds.append(bond_feat)
            graph.edge_index.append((node_idx, dummy_idx))

    def _add_global_dummy(
        self,
        graph,
        dummy_feature: torch.Tensor,
        dummy_bond: List[float],
    ) -> None:
        """Add a single global dummy node."""
        dummy_idx = len(graph.f_atoms)
        graph.f_atoms.append(dummy_feature.tolist())
        graph.atom_origin_type.append(AtomOriginType.DUMMY)

        n_connections = (
            graph.n_atoms * 2
            if isinstance(graph, ConnectedPairGraph)
            else graph.n_atoms
        )

        for i in range(n_connections):
            self._connect_dummy_to_node(graph, dummy_idx, i, dummy_bond)

    def _add_reactant_product_dummies(
        self,
        graph,
        dummy_feature: torch.Tensor,
        dummy_bond: List[float],
    ) -> None:
        """Add separate dummy nodes for reactants and products."""
        # reactant dummy
        dummy_reactant_idx = len(graph.f_atoms)
        graph.f_atoms.append(dummy_feature.tolist())
        graph.atom_origin_type.append(AtomOriginType.DUMMY)

        # product dummy
        dummy_product_idx = len(graph.f_atoms)
        graph.f_atoms.append(dummy_feature.tolist())
        graph.atom_origin_type.append(AtomOriginType.DUMMY)

        # connect to respective parts
        for i in range(graph.n_atoms):
            self._connect_dummy_to_node(
                graph, dummy_reactant_idx, i, dummy_bond
            )
            self._connect_dummy_to_node(
                graph, dummy_product_idx, i + graph.n_atoms, dummy_bond
            )

        # connect dummies if specified
        if self.dummy_dummy_connection == "bidirectional":
            graph.f_bonds.extend([dummy_bond, dummy_bond])
            graph.edge_index.extend(
                [
                    (dummy_reactant_idx, dummy_product_idx),
                    (dummy_product_idx, dummy_reactant_idx),
                ]
            )
