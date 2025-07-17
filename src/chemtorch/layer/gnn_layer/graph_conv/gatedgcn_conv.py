from typing import Any, Callable, Dict, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.resolver import activation_resolver
from torch_scatter import scatter

class GatedGCNConv(MessagePassing):
    """
    GatedGCN layer
    Residual Gated Graph ConvNets
    https://arxiv.org/pdf/1711.07553.pdf
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout,
        use_residual: bool = True,
        equivstable_pe = False,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        # **kwargs,
    ):
        super(GatedGCNConv, self).__init__()
        self.activation = activation_resolver(act, **(act_kwargs or {}))
        self.A = pyg_nn.Linear(in_channels, out_channels, bias=True)
        self.B = pyg_nn.Linear(in_channels, out_channels, bias=True)
        self.C = pyg_nn.Linear(in_channels, out_channels, bias=True)
        self.D = pyg_nn.Linear(in_channels, out_channels, bias=True)
        self.E = pyg_nn.Linear(in_channels, out_channels, bias=True)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        self.EquivStablePE = equivstable_pe
        if self.EquivStablePE:
            self.mlp_r_ij = nn.Sequential(
                nn.Linear(1, out_channels),
                self.activation,
                nn.Linear(out_channels, 1),
                nn.Sigmoid(),
            )

        self.bn_node_x = nn.BatchNorm1d(out_channels)
        self.bn_edge_e = nn.BatchNorm1d(out_channels)
        self.act_fn_x = self.activation
        self.act_fn_e = self.activation
        self.dropout = dropout
        self.use_residual = use_residual
        self.e = None

    def forward(self, batch):
        x, e, edge_index = batch.x, batch.edge_attr, batch.edge_index

        """
        x               : [n_nodes, in_dim]
        e               : [n_edges, in_dim]
        edge_index      : [2, n_edges]
        """
        if self.use_residual:
            x_in = x
            e_in = e

        Ax = self.A(x)
        Bx = self.B(x)
        Ce = self.C(e)
        Dx = self.D(x)
        Ex = self.E(x)

        # # Handling for Equivariant and Stable PE using LapPE
        # # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        pe_LapPE = batch.pe_EquivStableLapPE if self.EquivStablePE else None

        x, e = self.propagate(
            edge_index, Bx=Bx, Dx=Dx, Ex=Ex, Ce=Ce, e=e, Ax=Ax, PE=pe_LapPE
        )

        x = self.bn_node_x(x)
        e = self.bn_edge_e(e)

        x = self.act_fn_x(x)
        e = self.act_fn_e(e)

        x = F.dropout(x, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        if self.use_residual:
            x = x_in + x
            e = e_in + e

        batch.x = x
        batch.edge_attr = e

        return batch

    def message(self, Dx_i, Ex_j, PE_i, PE_j, Ce):
        """
        {}x_i           : [n_edges, out_dim]
        {}x_j           : [n_edges, out_dim]
        {}e             : [n_edges, out_dim]
        """
        e_ij = Dx_i + Ex_j + Ce
        sigma_ij = torch.sigmoid(e_ij)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        if self.EquivStablePE:
            r_ij = ((PE_i - PE_j) ** 2).sum(dim=-1, keepdim=True)
            r_ij = self.mlp_r_ij(
                r_ij
            )  # the MLP is 1 dim --> hidden_dim --> 1 dim
            sigma_ij = sigma_ij * r_ij

        self.e = e_ij
        return sigma_ij

    def aggregate(self, sigma_ij, index, Bx_j, Bx):
        """
        sigma_ij        : [n_edges, out_dim]  ; is the output from message() function
        index           : [n_edges]
        {}x_j           : [n_edges, out_dim]
        """
        dim_size = Bx.shape[0]  # or None ??   <--- Double check this

        sum_sigma_x = sigma_ij * Bx_j
        numerator_eta_xj = scatter(
            sum_sigma_x, index, 0, None, dim_size, reduce="sum"
        )

        sum_sigma = sigma_ij
        denominator_eta_xj = scatter(
            sum_sigma, index, 0, None, dim_size, reduce="sum"
        )

        out = numerator_eta_xj / (denominator_eta_xj + 1e-6)
        return out

    def update(self, aggr_out, Ax):
        """
        aggr_out        : [n_nodes, out_dim] ; is the output from aggregate() function after the aggregation
        {}x             : [n_nodes, out_dim]
        """
        x = Ax + aggr_out
        e_out = self.e
        del self.e
        return x, e_out
