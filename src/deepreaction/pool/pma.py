# started from code from https://github.com/pyg-team/pytorch_geometric?tab=readme-ov-file, MIT License, Copyright (c) 2023 PyG Team <team@pyg.org>

from typing import Optional

import torch
from torch import nn
from torch.nn import LayerNorm, Linear, MultiheadAttention, Parameter
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch


class MultiheadAttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        heads: int = 1,
        layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.channels = channels
        self.heads = heads
        self.dropout = dropout

        self.attn = MultiheadAttention(
            channels,
            heads,
            batch_first=True,
            dropout=dropout,
        )
        self.lin = Linear(channels, channels)
        self.layer_norm1 = LayerNorm(channels) if layer_norm else None
        self.layer_norm2 = LayerNorm(channels) if layer_norm else None

    def reset_parameters(self):
        self.attn._reset_parameters()
        self.lin.reset_parameters()
        if self.layer_norm1 is not None:
            self.layer_norm1.reset_parameters()
        if self.layer_norm2 is not None:
            self.layer_norm2.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        y_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """"""  # noqa: D419
        if y_mask is not None:
            y_mask = ~y_mask

        out, _ = self.attn(x, y, y, y_mask, need_weights=False)

        if x_mask is not None:
            out[~x_mask] = 0.0

        out = out + x

        if self.layer_norm1 is not None:
            out = self.layer_norm1(out)

        out = out + self.lin(out).relu()

        if self.layer_norm2 is not None:
            out = self.layer_norm2(out)

        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.channels}, "
            f"heads={self.heads}, "
            f"layer_norm={self.layer_norm1 is not None}, "
            f"dropout={self.dropout})"
        )


class SetAttentionBlock(torch.nn.Module):
    r"""The Set Attention Block (SAB) from the `"Set Transformer: A
    Framework for Attention-based Permutation-Invariant Neural Networks"
    <https://arxiv.org/abs/1810.00825>`_ paper.

    .. math::

        \mathrm{SAB}(\mathbf{X}) = \mathrm{MAB}(\mathbf{x}, \mathbf{y})

    Args:
        channels (int): Size of each input sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        norm (str, optional): If set to :obj:`False`, will not apply layer
            normalization. (default: :obj:`True`)
        dropout (float, optional): Dropout probability of attention weights.
            (default: :obj:`0`)
    """

    def __init__(
        self,
        channels: int,
        heads: int = 1,
        layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mab = MultiheadAttentionBlock(
            channels, heads, layer_norm, dropout
        )

    def reset_parameters(self):
        self.mab.reset_parameters()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.mab(x, x, mask, mask)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.mab.channels}, "
            f"heads={self.mab.heads}, "
            f"layer_norm={self.mab.layer_norm1 is not None}, "
            f"dropout={self.mab.dropout})"
        )


class PMA(nn.Module):
    def __init__(
        self,
        channels: int,
        num_seed_points: int = 1,
        heads: int = 1,
        layer_norm: bool = True,
        dropout: float = 0.0,
        num_decoder_blocks: int = 0,
    ):
        super().__init__()
        self.lin = nn.Linear(channels, channels)
        self.seed = nn.Parameter(torch.empty(1, num_seed_points, channels))
        self.mab = MultiheadAttentionBlock(
            channels, heads, layer_norm, dropout
        )
        self.decoders = torch.nn.ModuleList(
            [
                SetAttentionBlock(channels, heads, layer_norm, dropout)
                for _ in range(num_decoder_blocks)
            ]
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        torch.nn.init.xavier_uniform_(self.seed)
        self.mab.reset_parameters()
        for decoder in self.decoders:
            decoder.reset_parameters()

    def forward(self, batch: Batch) -> torch.Tensor:
        x, mask = to_dense_batch(batch.x, batch.batch)
        x = self.lin(x).relu()
        x = self.mab(self.seed.expand(x.size(0), -1, -1), x, y_mask=mask)

        for decoder in self.decoders:
            x = decoder(x)

        return x.mean(dim=1)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.seed.size(2)}, "
            f"num_seed_points={self.seed.size(1)}, "
            f"heads={self.mab.heads}, "
            f"layer_norm={self.mab.layer_norm1 is not None}, "
            f"dropout={self.mab.dropout})"
        )
