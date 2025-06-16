from torch import nn
from torch_geometric.data import Batch


class LinearEdgeEncoder(nn.Module):
    """
    Linear encoder for edge features.
    This encoder applies a linear transformation to edge features
    to project them into a specified output space.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
    ):
        """
        Initializes the LinearEdgeEncoder.
        Args:
            in_channels (int): Number of input edge features.
            out_channels (int): Number of output edge features.
            bias (bool): Whether to include a bias term in the linear transformation.
        """
        super(LinearEdgeEncoder, self).__init__()

        self.encoder = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, batch: Batch) -> Batch:
        batch.edge_attr = self.encoder(batch.edge_attr)
        return batch
