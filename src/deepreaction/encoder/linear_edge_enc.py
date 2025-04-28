from torch import nn


class LinearEdgeEncoder(nn.Module):
    """Linear encoder for edge features in a graph."""

    def __init__(
        self,
        in_channels,
        out_channels,
        bias=True,
    ):
        """Initialize the linear edge encoder.

        Parameters
        ----------
        in_channels : int
            The number of input channels (edge feature dimensions).
        out_channels : int
            The number of output channels for the encoded edge features.
        bias : bool, optional
            Whether to include a bias term in the linear transformation, by default True.

        """
        super().__init__()

        self.encoder = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, batch):
        batch.edge_attr = self.encoder(batch.edge_attr)
        return batch
