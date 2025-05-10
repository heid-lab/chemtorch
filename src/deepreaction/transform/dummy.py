from typing import Dict, Optional

import torch
from torch import nn
from torch_geometric.data import Data

from deepreaction.representation.graph_representations.graph_reprs_utils import (
    AtomOriginType,
    EdgeOriginType,
)


class DummyNodeTransform(nn.Module):
    def __init__(
        self,
        mode: str,
        connection_type: Optional[str] = "to_dummy",
        dummy_dummy_connection: Optional[str] = None,
        feature_init: str = "zeros",
        type: str = "graph",
    ):
        super(DummyNodeTransform, self).__init__()
        self.mode = mode
        self.connection_type = connection_type
        self.dummy_dummy_connection = dummy_dummy_connection
        self.feature_init = feature_init
        self.dummy_origin_type = 2

    def forward(self, x: Data) -> Data:
        if (
            AtomOriginType.REACTANT_PRODUCT.value in x.atom_origin_type
            and self.mode == "reactant_product"
        ):
            raise ValueError("CGR not supported with reactant/product dummies")

        node_feat_dim = x.x.size(1)
        feature_init_fn = (
            torch.zeros if self.feature_init == "zeros" else torch.ones
        )

        # Create dummy features matching existing dimensions
        dummy_feature = feature_init_fn(1, node_feat_dim, device=x.x.device)
        dummy_bond = None

        if hasattr(x, "edge_attr"):
            bond_feat_dim = x.edge_attr.size(1)
            dummy_bond = feature_init_fn(
                1, bond_feat_dim, device=x.edge_attr.device
            )

        if self.mode == "global":
            self._add_global_dummy(x, dummy_feature, dummy_bond)
        elif self.mode == "reactant_product":
            self._add_reactant_product_dummies(x, dummy_feature, dummy_bond)
        else:
            raise ValueError(f"Invalid dummy mode: {self.mode}")
        return x

    def _get_pretransform_attributes(
        self, data: Data
    ) -> Dict[str, torch.Tensor]:
        return {
            k: v
            for k, v in data
            if k
            not in {
                "x",
                "edge_index",
                "edge_attr",
                "y",
                "smiles",
                "atom_origin_type",
                "batch",
                "atom_compound_idx",
            }
            and isinstance(v, torch.Tensor)
        }

    def _create_dummy_features(
        self, attr: torch.Tensor, num_dummies: int
    ) -> torch.Tensor:
        shape = list(attr.shape)
        shape[0] = num_dummies
        return (
            torch.zeros(*shape, dtype=attr.dtype, device=attr.device)
            if self.feature_init == "zeros"
            else torch.ones(*shape, dtype=attr.dtype, device=attr.device)
        )

    def _connect_dummy_to_node(
        self,
        data: Data,
        dummy_idx: int,
        node_idx: int,
        dummy_bond: Optional[torch.Tensor],
    ):
        new_edges = []
        new_edge_attrs = []
        new_origin_types = []

        # Handle from_dummy connections
        if self.connection_type in ["bidirectional", "from_dummy"]:
            new_edges.append(
                torch.tensor(
                    [[dummy_idx], [node_idx]], device=data.edge_index.device
                )
            )
            new_origin_types.append(self.dummy_origin_type)
            if dummy_bond is not None:
                new_edge_attrs.append(dummy_bond)

        # Handle to_dummy connections
        if self.connection_type in ["bidirectional", "to_dummy"]:
            new_edges.append(
                torch.tensor(
                    [[node_idx], [dummy_idx]], device=data.edge_index.device
                )
            )
            new_origin_types.append(self.dummy_origin_type)
            if dummy_bond is not None:
                new_edge_attrs.append(dummy_bond)

        # Update data
        if new_edges:
            data.edge_index = torch.cat([data.edge_index] + new_edges, dim=1)
            if hasattr(data, "edge_attr") and dummy_bond is not None:
                data.edge_attr = torch.cat(
                    [data.edge_attr] + new_edge_attrs, dim=0
                )
            if hasattr(data, "edge_origin_type"):
                data.edge_origin_type = torch.cat(
                    [
                        data.edge_origin_type,
                        torch.tensor(
                            new_origin_types,
                            dtype=torch.long,
                            device=data.edge_origin_type.device,
                        ),
                    ]
                )

    def _add_global_dummy(
        self,
        data: Data,
        dummy_feature: torch.Tensor,
        dummy_bond: Optional[torch.Tensor],
    ):
        # Add dummy node
        data.x = torch.cat([data.x, dummy_feature], dim=0)
        dummy_idx = data.x.size(0) - 1

        # Connect to all existing nodes
        for node_idx in range(data.x.size(0) - 1):
            self._connect_dummy_to_node(data, dummy_idx, node_idx, dummy_bond)

        # Update origin tracking
        data.atom_origin_type = torch.cat(
            [
                data.atom_origin_type,
                torch.tensor(
                    [AtomOriginType.DUMMY.value],
                    device=data.atom_origin_type.device,
                ),
            ]
        )
        if hasattr(data, "atom_compound_idx"):
            data.atom_compound_idx = torch.cat(
                [
                    data.atom_compound_idx,
                    torch.tensor([-1], device=data.atom_compound_idx.device),
                ]
            )

        # Extend pre-transform attributes
        for attr_name, attr_value in self._get_pretransform_attributes(
            data
        ).items():
            dummy_attr = self._create_dummy_features(attr_value, 1)
            setattr(
                data, attr_name, torch.cat([attr_value, dummy_attr], dim=0)
            )

    def _add_reactant_product_dummies(
        self,
        data: Data,
        dummy_feature: torch.Tensor,
        dummy_bond: Optional[torch.Tensor],
    ):
        # Add two dummy nodes
        data.x = torch.cat([data.x, dummy_feature, dummy_feature], dim=0)
        dummy_reactant_idx = data.x.size(0) - 2
        dummy_product_idx = data.x.size(0) - 1

        # Get reactant/product indices
        reac_mask = data.atom_origin_type == AtomOriginType.REACTANT
        prod_mask = data.atom_origin_type == AtomOriginType.PRODUCT
        reac_indices = torch.where(reac_mask)[0].tolist()
        prod_indices = torch.where(prod_mask)[0].tolist()

        # Connect dummies to their respective nodes
        for idx in reac_indices:
            self._connect_dummy_to_node(
                data, dummy_reactant_idx, idx, dummy_bond
            )
        for idx in prod_indices:
            self._connect_dummy_to_node(
                data, dummy_product_idx, idx, dummy_bond
            )

        # Connect dummies to each other if needed
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
            if hasattr(data, "edge_origin_type"):
                data.edge_origin_type = torch.cat(
                    [
                        data.edge_origin_type,
                        torch.tensor(
                            [self.dummy_origin_type] * 2,
                            device=data.edge_origin_type.device,
                        ),
                    ]
                )

        # Update origin tracking
        data.atom_origin_type = torch.cat(
            [
                data.atom_origin_type,
                torch.tensor(
                    [AtomOriginType.DUMMY.value] * 2,
                    device=data.atom_origin_type.device,
                ),
            ]
        )
        if hasattr(data, "atom_compound_idx"):
            data.atom_compound_idx = torch.cat(
                [
                    data.atom_compound_idx,
                    torch.tensor(
                        [-1, -1], device=data.atom_compound_idx.device
                    ),
                ]
            )

        # Extend pre-transform attributes
        for attr_name, attr_value in self._get_pretransform_attributes(
            data
        ).items():
            dummy_attr = self._create_dummy_features(attr_value, 2)
            setattr(
                data, attr_name, torch.cat([attr_value, dummy_attr], dim=0)
            )
