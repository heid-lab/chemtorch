from torch import nn
from torch_geometric.data import Batch

from chemtorch.encoder.linear_edge_enc import LinearEdgeEncoder
from chemtorch.encoder.linear_node_enc import LinearNodeEncoder


class LinearEncoder(nn.Module):
    """
    Linear encoder for node and edge features.
    This encoder applies a linear transformation to both node and edge features
    to project them into a common hidden space.
    """

    def __init__(
        self,
        node_encoder_in_channels: int,
        edge_encoder_in_channels: int,
        out_channels: int,
        bias: bool = True,
    ):
        """
        Initializes the LinearEncoder.
        Args:
            node_encoder_in_channels (int): Number of input node features.
            edge_encoder_in_channels (int): Number of input edge features.
            out_channels (int): Number of output channels for the linear transformation.
            bias (bool): Whether to include a bias term in the linear transformations.
        """
        super(LinearEncoder, self).__init__()

        self.node_encoder = LinearNodeEncoder(
            in_channels=node_encoder_in_channels,
            out_channels=out_channels,
            bias=bias,
        )
        self.edge_encoder = LinearEdgeEncoder(
            in_channels=edge_encoder_in_channels,
            out_channels=out_channels,
            bias=bias,
        )

    def forward(self, batch: Batch) -> Batch:
        batch = self.node_encoder(batch)
        batch = self.edge_encoder(batch)
        return batch
