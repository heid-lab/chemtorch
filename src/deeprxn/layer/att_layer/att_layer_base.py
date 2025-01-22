from abc import abstractmethod

from torch import nn
from torch_geometric.data import Batch


class AttLayer(nn.Module):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, batch: Batch) -> Batch:
        pass
