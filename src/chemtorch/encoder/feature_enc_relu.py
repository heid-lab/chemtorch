from torch import nn
from torch_geometric.data import Batch
import torch
import torch.nn.functional as F


class FeatureEncoderRelu(nn.Module):

    def __init__(
        self,
        in_channels: int,
        feature_out_channels: int = 110,
        hidden_channels: int = 128,
        dropout: float = 0.1,
        out_channels: int = 88
    ):
        super(FeatureEncoderRelu, self).__init__()

        # self.feature_encoder = nn.Linear(feature_in_channels, feature_out_channels)
        self.feature_encoder = nn.Sequential(
            nn.Linear(256, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, feature_out_channels),
        )

        self.edge_init = nn.Linear(220, 110)


    def forward(self, batch: Batch) -> Batch:
        print("WE ARE HERE", batch.x.shape, batch.extra_atom_features.shape)
        print(batch.extra_atom_features)
        batch.extra_atom_features = self.feature_encoder(batch.extra_atom_features)

        print("UND DANACH", batch.x.shape, batch.extra_atom_features.shape)

        batch.x = torch.cat([batch.x, batch.extra_atom_features], dim=1)


        row, col = batch.edge_index
        print("WIR SIND DAHINTER", batch.x.shape, batch.edge_attr.shape, batch.extra_atom_features.shape)
        batch.h_0 = F.relu(
            self.edge_init(torch.cat([batch.x[row], batch.edge_attr], dim=1))
        )
        batch.h = batch.h_0

        print("WIR SIND DURCH", batch.h_0.shape, batch.h.shape, batch.x.shape)

        return batch