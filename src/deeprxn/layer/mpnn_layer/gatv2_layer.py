import torch
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv

from deeprxn.layer.mpnn_layer.mpnn_layer_base import MPNNLayerBase


class GATv2Layer(MPNNLayerBase):
    """Graph Attention Network Layer wrapper."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        dropout: float = 0.0,
        concat: bool = True,
        use_edge_attr: bool = True,
    ):
        """Initialize GAT layer.

        Args:
            in_channels: Number of input features
            out_channels: Number of output features
            heads: Number of attention heads
            dropout: Dropout probability
            concat: Whether to concatenate or average multi-head attention outputs
        """
        super().__init__(in_channels, out_channels)

        self.use_edge_attr = use_edge_attr

        self.gat = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            concat=concat,
            edge_dim=in_channels if use_edge_attr else None,
        )

    def forward(self, batch: Batch) -> Batch:
        """Forward pass of the GAT layer.

        Args:
            batch: PyG batch containing:
                - x: Node features
                - edge_index: Graph connectivity

        Returns:
            Updated batch with new node features
        """
        if self.use_edge_attr:
            batch.x = self.gat(batch.x, batch.edge_index, batch.edge_attr)
        else:
            batch.x = self.gat(batch.x, batch.edge_index)
        return batch
