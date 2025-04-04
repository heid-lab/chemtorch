import torch
from torch import nn
from torch_geometric.data import Batch

from deeprxn.encoder.encoder_base import Encoder


class LinearQMFEncoder(Encoder):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        as_variable: bool = True,
        do_both: bool = False,
        both_hidden: int = 128,
    ):
        super().__init__()
        self.as_variable = as_variable
        self.encoder = nn.Linear(in_channels, out_channels, bias=bias)
        self.do_both = do_both
        if self.do_both:
            self.encoder_hidden = nn.Linear(in_channels, both_hidden, bias=bias)

    def forward(self, batch: Batch) -> Batch:
        encoded = self.encoder(batch.qm_f)

        if self.do_both:
            encoded_hidden = self.encoder_hidden(batch.qm_f)
            batch.qm_f = encoded_hidden
        elif self.as_variable:
            batch.qm_f = encoded

        if not self.as_variable or self.do_both:
            batch.x = torch.cat([batch.x, encoded], dim=1)

        return batch
