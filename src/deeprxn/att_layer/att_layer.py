from abc import abstractmethod

from torch import nn
from torch_geometric.data import Batch


class AttLayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    @abstractmethod
    def forward(self, batch: Batch) -> Batch:
        pass
