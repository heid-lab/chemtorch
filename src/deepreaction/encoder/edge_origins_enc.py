import torch
from torch import nn


class EdgeOriginsEncoder(nn.Module):
    """Edge origins encoder for graph neural networks."""

    def __init__(
        self,
        in_channels,
        out_channels,
        as_variable=False,
    ):
        """Initialize the edge origins encoder.

        Parameters
        ----------
        in_channels : int
            The input dimension of edge origin encodings.
        out_channels : int
            The output dimension after encoding.
        as_variable : bool, optional
            Whether to store output as a separate variable or concatenate 
            to edge attributes, by default False.

        """
        super().__init__()
        self.as_variable = as_variable
        self.raw_norm = nn.BatchNorm1d(in_channels)
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, batch):
        if not hasattr(batch, "edge_origin_encoding"):
            raise ValueError(
                "Batch object does not have edge_origin_encoding attribute"
            )

        edge_origin_encoding = getattr(batch, "edge_origin_encoding")

        edge_origin_encoding = self.raw_norm(edge_origin_encoding)
        edge_origin_encoding = self.linear(edge_origin_encoding)

        if self.as_variable:
            batch.edge_origin_encoding = edge_origin_encoding
        else:
            batch.edge_attr = torch.cat(
                [batch.edge_attr, edge_origin_encoding], dim=1
            )

        return batch
