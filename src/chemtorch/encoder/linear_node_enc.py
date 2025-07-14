from torch import nn
from torch_geometric.data import Batch


class LinearNodeEncoder(nn.Module):
    """
    Linear encoder for node features.
    This encoder applies a linear transformation to node features
    to project them into a specified hidden space.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
    ):
        """
        Initializes the LinearNodeEncoder.
        Args:
            in_channels (int): Number of input node features.
            out_channels (int): Number of output channels for the linear transformation.
            bias (bool): Whether to include a bias term in the linear transformation.
        """
        super(LinearNodeEncoder, self).__init__()

        self.encoder = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, batch: Batch) -> Batch:
        batch.x = self.encoder(batch.x)
        return batch
