import torch.nn as nn

from deepreaction.act.act import Activation


class FFNHead(nn.Module):
   """Feed forward network head with configurable layers."""

   def __init__(
       self,
       in_channels,
       out_channels,
       hidden_channels,
       num_layers=2,
       dropout=0.02,
       activation="relu",
   ):
       """Initialize the feed forward network head.

       Parameters
       ----------
       in_channels : int
           The input feature dimension.
       out_channels : int
           The output feature dimension.
       hidden_channels : int
           The hidden layer dimension.
       num_layers : int, optional
           The number of linear layers, by default 2.
       dropout : float, optional
           Dropout probability, by default 0.02.
       activation : str, optional
           Activation function type, by default "relu".

       """
       super().__init__()
       self.activation = Activation(activation_type=activation)

       layers = []
       current_dim = in_channels

       for _ in range(num_layers - 1):
           layers.extend(
               [
                   nn.Dropout(dropout),
                   nn.Linear(current_dim, hidden_channels),
                   self.activation,
               ]
           )
           current_dim = hidden_channels

       layers.extend(
           [nn.Dropout(dropout), nn.Linear(current_dim, out_channels)]
       )

       self.ffn = nn.Sequential(*layers)

   def forward(self, batch):
       return self.ffn(batch.x).squeeze(-1)
