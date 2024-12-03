from typing import List, Optional

import torch
from torch_geometric.data import Data

from deeprxn.representation.rxn_graph import AtomOriginType
from deeprxn.transform.transform_base import TransformBase


class DummyNodeTransform(TransformBase):
    def __init__(
        self,
        mode: str,
        connection_type: Optional[str] = "to_dummy",
        dummy_dummy_connection: Optional[str] = None,
        feature_init: str = "zeros",
    ):
        self.mode = mode
        self.connection_type = connection_type
        self.dummy_dummy_connection = dummy_dummy_connection
        self.feature_init = feature_init

    def forward(self, data: Data) -> Data:
        """Apply dummy node transform to the graph."""

        node_feat_dim = data.x.size(1)
        feature_init_fn = (
            torch.zeros if self.feature_init == "zeros" else torch.ones
        )
        dummy_feature = feature_init_fn(1, node_feat_dim, device=data.x.device)

        dummy_bond = None
        if hasattr(data, "edge_attr"):
            dummy_bond = feature_init_fn(
                1, data.edge_attr.size(1), device=data.edge_attr.device
            )

        if self.mode == "global":
            self._add_global_dummy(data, dummy_feature, dummy_bond)
        elif self.mode == "reactant_product":
            self._add_reactant_product_dummies(data, dummy_feature, dummy_bond)
        else:
            raise ValueError(f"Unknown dummy node mode: {self.mode}")

        return data

    def _connect_dummy_to_node(
        self,
        data: Data,
        dummy_idx: int,
        node_idx: int,
        dummy_bond: Optional[torch.Tensor],
    ) -> None:
        new_edges = []
        new_features = []

        if self.connection_type in ["bidirectional", "from_dummy"]:
            new_edges.append(
                torch.tensor(
                    [[dummy_idx], [node_idx]], device=data.edge_index.device
                )
            )
            if hasattr(data, "edge_attr") and dummy_bond is not None:
                new_features.append(dummy_bond)

        if self.connection_type in ["bidirectional", "to_dummy"]:
            new_edges.append(
                torch.tensor(
                    [[node_idx], [dummy_idx]], device=data.edge_index.device
                )
            )
            if hasattr(data, "edge_attr") and dummy_bond is not None:
                new_features.append(dummy_bond)

        if new_edges:
            data.edge_index = torch.cat([data.edge_index, *new_edges], dim=1)
            if hasattr(data, "edge_attr") and dummy_bond is not None:
                data.edge_attr = torch.cat(
                    [data.edge_attr, *new_features], dim=0
                )

    def _add_global_dummy(
        self,
        data: Data,
        dummy_feature: torch.Tensor,
        dummy_bond: Optional[torch.Tensor],
    ) -> None:
        """Add a single global dummy node."""
        data.x = torch.cat([data.x, dummy_feature], dim=0)
        dummy_idx = data.x.size(0) - 1

        for i in range(dummy_idx):
            self._connect_dummy_to_node(data, dummy_idx, i, dummy_bond)

        data.atom_origin_type = torch.cat(
            [
                data.atom_origin_type,
                torch.tensor(
                    [AtomOriginType.DUMMY], device=data.atom_origin_type.device
                ),
            ]
        )

    def _add_reactant_product_dummies(
        self,
        data: Data,
        dummy_feature: torch.Tensor,
        dummy_bond: Optional[torch.Tensor],
    ) -> None:
        n_atoms = (
            data.x.size(0) // 2
        )  # assuming equal split between reactants and products

        data.x = torch.cat([data.x, dummy_feature, dummy_feature], dim=0)
        dummy_reactant_idx = data.x.size(0) - 2
        dummy_product_idx = data.x.size(0) - 1

        for i in range(n_atoms):
            self._connect_dummy_to_node(
                data, dummy_reactant_idx, i, dummy_bond
            )
            self._connect_dummy_to_node(
                data, dummy_product_idx, i + n_atoms, dummy_bond
            )

        if self.dummy_dummy_connection == "bidirectional":
            dummy_edges = torch.tensor(
                [
                    [dummy_reactant_idx, dummy_product_idx],
                    [dummy_product_idx, dummy_reactant_idx],
                ],
                device=data.edge_index.device,
            ).t()
            data.edge_index = torch.cat([data.edge_index, dummy_edges], dim=1)
            if hasattr(data, "edge_attr") and dummy_bond is not None:
                data.edge_attr = torch.cat(
                    [data.edge_attr, dummy_bond.repeat(2, 1)], dim=0
                )

        data.atom_origin_type = torch.cat(
            [
                data.atom_origin_type,
                torch.tensor(
                    [AtomOriginType.DUMMY, AtomOriginType.DUMMY],
                    device=data.atom_origin_type.device,
                ),
            ]
        )
