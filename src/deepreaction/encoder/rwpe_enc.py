import torch
from torch import nn


class RWEncoder(nn.Module):
    """Random walk positional encoder for graph neural networks."""

    def __init__(
        self,
        in_channels,
        out_channels,
        transform_type="normal",
        as_variable=False,
    ):
        """Initialize the random walk positional encoder.

        Parameters
        ----------
        in_channels : int
            The input dimension of random walk features.
        out_channels : int
            The output dimension of encoded positional features.
        transform_type : str, optional
            The transformation type for positional encodings, by default "normal".
            Options: "normal", "difference", or "reactant_difference".
        as_variable : bool, optional
            Whether to store as a separate variable or concatenate to node features, by default False.

        """
        super().__init__()
        self.as_variable = as_variable

        in_channels = (
            in_channels // 2 if transform_type == "difference" else in_channels
        )

        self.raw_norm = nn.BatchNorm1d(in_channels)
        self.transform_type = transform_type

        self.pe_encoder = nn.Linear(in_channels, out_channels)

    def forward(self, batch):
        if not hasattr(batch, "randomwalkpe"):
            raise ValueError(
                "Batch object does not have randomwalkpe attribute"
            )

        pos_enc = getattr(batch, "randomwalkpe")

        if self.transform_type == "reactant_difference":
            mid = pos_enc.shape[-1] // 2
            reactant, product = pos_enc[..., :mid], pos_enc[..., mid:]
            pos_enc = torch.cat([reactant, product - reactant], dim=-1)
        if self.transform_type == "difference":
            mid = pos_enc.shape[-1] // 2
            reactant, product = pos_enc[..., :mid], pos_enc[..., mid:]
            pos_enc = product - reactant

        pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.pe_encoder(pos_enc)
        # h = self.linear_x(batch.x)
        if self.as_variable:
            batch.randomwalkpe = pos_enc
        else:
            batch.x = torch.cat([batch.x, pos_enc], dim=1)

        return batch
