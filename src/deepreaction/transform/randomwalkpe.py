import torch
import torch_geometric
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import (
    get_self_loop_attr,
    is_torch_sparse_tensor,
    scatter,
    to_edge_index,
    to_torch_coo_tensor,
    to_torch_csr_tensor,
)

from deepreaction.transform.transform_base import TransformBase


class RandomWalkPETransform(TransformBase):
    """
    This code includes implementations adapted from PyTorch Geometric
    (https://github.com/pyg-team/pytorch_geometric)
    # TODO: check out how to cite code
    """

    def __init__(
        self,
        walk_length: int,
        attr_name=None,
        type: str = "graph",
    ) -> None:
        self.walk_length = walk_length
        self.attr_name = attr_name

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        row, col = data.edge_index
        N = data.num_nodes
        assert N is not None

        if data.edge_weight is None:
            value = torch.ones(data.num_edges, device=row.device)
        else:
            value = data.edge_weight
        value = scatter(value, row, dim_size=N, reduce="sum").clamp(min=1)[row]
        value = 1.0 / value

        if N <= 2_000:  # Dense code path for faster computation:
            adj = torch.zeros((N, N), device=row.device)
            adj[row, col] = value
            loop_index = torch.arange(N, device=row.device)
        elif torch_geometric.typing.NO_MKL:  # pragma: no cover
            adj = to_torch_coo_tensor(data.edge_index, value, size=data.size())
        else:
            adj = to_torch_csr_tensor(data.edge_index, value, size=data.size())

        def get_pe(out: Tensor) -> Tensor:
            if is_torch_sparse_tensor(out):
                return get_self_loop_attr(*to_edge_index(out), num_nodes=N)
            return out[loop_index, loop_index]

        out = adj
        pe_list = [get_pe(out)]
        for _ in range(self.walk_length - 1):
            out = out @ adj
            pe_list.append(get_pe(out))

        pe = torch.stack(pe_list, dim=-1)

        if self.attr_name is None:
            if data.x is not None:
                x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
                data.x = torch.cat([x, pe.to(x.device, x.dtype)], dim=-1)
            else:
                data.x = pe
        else:
            data[self.attr_name] = pe

        return data
