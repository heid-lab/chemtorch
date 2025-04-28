from torch import nn


class LinearNodeEncoder(nn.Module):
    """Linear node feature encoder for graph neural networks."""

    def __init__(
        self,
        in_channels,
        out_channels,
        bias=True,
    ):
        """Initialize the linear node encoder.

        Parameters
        ----------
        in_channels : int
            The dimension of input node features.
        out_channels : int
            The dimension of output node features.
        bias : bool, optional
            Whether to include a bias term, by default True.

        """
        super().__init__()

        self.encoder = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, batch):
        batch.x = self.encoder(batch.x)
        return batch
