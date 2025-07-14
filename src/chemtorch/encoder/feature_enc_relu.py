from torch import nn
from torch_geometric.data import Batch
import torch
import torch.nn.functional as F


class FeatureEncoderRelu(nn.Module):

    def __init__(
        self,
        feature_in_channels: int = 256,   # Length of the extra_atom_features vector
        feature_out_channels: int = 110,  # Whatever you want
        feature_hidden_channels: int = 128, # Whatever you want
        features_dropout: float = 0.1,  
        edge_embedding_size: int = 1200,  # Its the hidden_channels of the model (model.hidden_channels)
        modified_in_channels: int = 220,  # features_out_channels + num_node_features + num_edge_feature
    ):
        super(FeatureEncoderRelu, self).__init__()

        # self.feature_encoder = nn.Linear(feature_in_channels, feature_out_channels)
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_in_channels, feature_hidden_channels),
            nn.ReLU(),
            nn.Dropout(features_dropout),
            nn.Linear(feature_hidden_channels, feature_out_channels),
        )

        # Edge embeddings must match the hidden_channels of the model
        self.edge_init = nn.Linear(modified_in_channels, edge_embedding_size)  # features_out_channels + num_node_features + num_edge_features

    def forward(self, batch: Batch) -> Batch:
        batch.extra_atom_features = self.feature_encoder(batch.extra_atom_features)


        batch.x1 = torch.cat([batch.x, batch.extra_atom_features], dim=1)


        row, col = batch.edge_index
        batch.h_0 = F.relu(
            self.edge_init(torch.cat([batch.x1[row], batch.edge_attr], dim=1))
        )
        batch.h = batch.h_0

        return batch