import torch.nn as nn


class MSELoss(nn.Module):
   """Mean squared error loss wrapper."""

   def __init__(self, reduction="sum"):
       """Initialize the MSE loss.

       Parameters
       ----------
       reduction : str, optional
           Specifies the reduction to apply to the output, by default "sum".
           Options: 'none' | 'mean' | 'sum'.

       """
       super(MSELoss, self).__init__()
       self.mse = nn.MSELoss(reduction=reduction)

   def forward(self, preds, y):
       return self.mse(preds, y)
