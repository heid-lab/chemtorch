import math
import os
from multiprocessing.sharedctypes import Value

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul


class TransConv(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers=2,
        num_heads=1,
        alpha=0.5,
        dropout=0.5,
        use_bn=True,
        use_residual=True,
        use_weight=True,
        use_act=False,
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                TransConvLayer(
                    hidden_channels,
                    hidden_channels,
                    num_heads=num_heads,
                    use_weight=use_weight,
                )
            )
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.residual = use_residual
        self.alpha = alpha
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_weight = None
        layer_ = []

        # input MLP layer
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_.append(x)

        for i, conv in enumerate(self.convs):
            # graph convolution with full attention aggregation
            x = conv(x, x, edge_index, edge_weight)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        return x

    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  # [layer num, N, N]


class SGFormer(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        num_heads=1,
        alpha=0.5,
        dropout=0.5,
        use_bn=True,
        use_residual=True,
        use_weight=True,
        use_graph=True,
        use_act=False,
        graph_weight=0.8,
        gnn=None,
        aggregate="add",
    ):
        super().__init__()
        self.trans_conv = TransConv(
            in_channels,
            hidden_channels,
            num_layers,
            num_heads,
            alpha,
            dropout,
            use_bn,
            use_residual,
            use_weight,
        )
        self.gnn = gnn
        self.use_graph = use_graph
        self.graph_weight = graph_weight
        self.use_act = use_act

        self.aggregate = aggregate

        if aggregate == "add":
            self.fc = nn.Linear(hidden_channels, out_channels)
        elif aggregate == "cat":
            self.fc = nn.Linear(2 * hidden_channels, out_channels)
        else:
            raise ValueError(f"Invalid aggregate type:{aggregate}")

        self.params1 = list(self.trans_conv.parameters())
        self.params2 = (
            list(self.gnn.parameters()) if self.gnn is not None else []
        )
        self.params2.extend(list(self.fc.parameters()))

        self.pool = global_add_pool

        layers = [
            nn.Dropout(0.02),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.02),
            nn.Linear(hidden_channels, 1),
        ]
        self.ffn = nn.Sequential(*layers)

    def forward(self, data):
        x1 = self.trans_conv(data)
        # print(f"Dimention of x1: {x1.shape}")
        if self.use_graph:
            x2 = self.gnn(data)
            # print(f"Dimention of x2: {x2.shape}")
            if self.aggregate == "add":
                x = self.graph_weight * x2 + (1 - self.graph_weight) * x1
            else:
                x = torch.cat((x1, x2), dim=1)
        else:
            x = x1

        x = self.pool(x, data.batch)
        x = self.ffn(x).squeeze(-1)
        return x

    def get_attentions(self, x):
        attns = self.trans_conv.get_attentions(x)  # [layer num, N, N]

        return attns

    def reset_parameters(self):
        self.trans_conv.reset_parameters()
        if self.use_graph:
            self.gnn.reset_parameters()


from operator import truediv

import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse
from torch import FloatTensor
from torch_geometric.nn import (
    APPNP,
    GATConv,
    GCNConv,
    JumpingKnowledge,
    MessagePassing,
    SGConv,
)
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul


class GCN(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        dropout=0.5,
        save_mem=True,
        use_bn=True,
    ):
        super(GCN, self).__init__()

        self.convs = nn.ModuleList()
        # self.convs.append(
        #     GCNConv(in_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=not save_mem)
        )

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            # self.convs.append(
            #     GCNConv(hidden_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=not save_mem)
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # self.convs.append(
        #     GCNConv(hidden_channels, out_channels, cached=not save_mem, normalize=not save_mem))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=not save_mem)
        )

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_weight = None
        for i, conv in enumerate(self.convs[:-1]):
            if edge_weight is None:
                x = conv(x, edge_index)
            else:
                x = conv(x, edge_index, edge_weight)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, data.edge_index)
        return x


class SGC(nn.Module):
    def __init__(self, in_channels, out_channels, hops):
        """takes 'hops' power of the normalized adjacency"""
        super(SGC, self).__init__()
        self.conv = SGConv(in_channels, out_channels, hops, cached=True)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, data):
        edge_index = data.graph["edge_index"]
        x = data.graph["node_feat"]
        x = self.conv(x, edge_index)
        return x


class MLP(nn.Module):
    """adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py"""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dropout=0.5,
    ):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x = data.graph["node_feat"]
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class GAT(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        dropout=0.5,
        use_bn=False,
        heads=2,
        out_heads=1,
    ):
        super(GAT, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(
                in_channels,
                hidden_channels,
                dropout=dropout,
                heads=heads,
                concat=True,
            )
        )

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_channels * heads,
                    hidden_channels,
                    dropout=dropout,
                    heads=heads,
                    concat=True,
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

        self.convs.append(
            GATConv(
                hidden_channels * heads,
                out_channels,
                dropout=dropout,
                heads=out_heads,
                concat=False,
            )
        )

        self.dropout = dropout
        self.activation = F.elu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x = data.graph["node_feat"]
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, data.graph["edge_index"])
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, data.graph["edge_index"])
        return x


class GCNJK(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        dropout=0.5,
        save_mem=False,
        jk_type="max",
    ):
        super(GCNJK, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(
                in_channels,
                hidden_channels,
                cached=not save_mem,
                normalize=not save_mem,
            )
        )

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(
                    hidden_channels,
                    hidden_channels,
                    cached=not save_mem,
                    normalize=not save_mem,
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            GCNConv(
                hidden_channels,
                hidden_channels,
                cached=not save_mem,
                normalize=not save_mem,
            )
        )

        self.dropout = dropout
        self.activation = F.relu
        self.jump = JumpingKnowledge(
            jk_type, channels=hidden_channels, num_layers=1
        )
        if jk_type == "cat":
            self.final_project = nn.Linear(
                hidden_channels * num_layers, out_channels
            )
        else:  # max or lstm
            self.final_project = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.jump.reset_parameters()
        self.final_project.reset_parameters()

    def forward(self, data):
        x = data.graph["node_feat"]
        xs = []
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, data.graph["edge_index"])
            x = self.bns[i](x)
            x = self.activation(x)
            xs.append(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, data.graph["edge_index"])
        xs.append(x)
        x = self.jump(xs)
        x = self.final_project(x)
        return x


class APPNP_Net(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        dropout=0.5,
        K=10,
        alpha=0.1,
    ):
        super(APPNP_Net, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.prop1 = APPNP(K, alpha)
        self.dropout = dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x = data.graph["node_feat"]
        edge_index = data.graph["edge_index"]
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return x


class H2GCN(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int,
        class_dim: int,
        k: int = 2,
        dropout: float = 0.5,
        relu: bool = True,
    ):
        super(H2GCN, self).__init__()
        self.dropout = dropout
        self.k = k
        self.w_embed = nn.Parameter(
            torch.zeros(size=(feat_dim, hidden_dim)), requires_grad=True
        )
        self.w_classify = nn.Parameter(
            torch.zeros(
                size=((2 ** (self.k + 1) - 1) * hidden_dim, class_dim)
            ),
            requires_grad=True,
        )
        self.params = [self.w_embed, self.w_classify]
        self.initialized = False
        self.a1 = None
        self.a2 = None
        self.relu = relu
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_embed)
        nn.init.xavier_uniform_(self.w_classify)

    @staticmethod
    def _indicator(sp_tensor: torch.sparse.Tensor) -> torch.sparse.Tensor:
        csp = sp_tensor.coalesce()
        return torch.sparse_coo_tensor(
            indices=csp.indices(),
            values=torch.where(csp.values() > 0, 1, 0),
            size=csp.size(),
            dtype=torch.float,
        )

    @staticmethod
    def _spspmm(
        sp1: torch.sparse.Tensor, sp2: torch.sparse.Tensor
    ) -> torch.sparse.Tensor:
        assert (
            sp1.shape[1] == sp2.shape[0]
        ), "Cannot multiply size %s with %s" % (sp1.shape, sp2.shape)
        sp1, sp2 = sp1.coalesce(), sp2.coalesce()
        index1, value1 = sp1.indices(), sp1.values()
        index2, value2 = sp2.indices(), sp2.values()
        m, n, k = sp1.shape[0], sp1.shape[1], sp2.shape[1]
        indices, values = torch_sparse.spspmm(
            index1, value1, index2, value2, m, n, k
        )
        return torch.sparse_coo_tensor(
            indices=indices, values=values, size=(m, k), dtype=torch.float
        )

    @classmethod
    def _adj_norm(cls, adj: torch.sparse.Tensor) -> torch.sparse.Tensor:
        n = adj.size(0)
        d_diag = torch.pow(torch.sparse.sum(adj, dim=1).values(), -0.5)
        d_diag = torch.where(
            torch.isinf(d_diag), torch.full_like(d_diag, 0), d_diag
        )
        d_tiled = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=d_diag,
            size=(n, n),
        )
        return cls._spspmm(cls._spspmm(d_tiled, adj), d_tiled)

    def _prepare_prop(self, edge_index, n):
        # convert edge index to torch sparse tensor
        adj = torch.sparse_coo_tensor(
            indices=edge_index.int(),
            values=torch.ones(edge_index.shape[1]).to(edge_index.device),
            size=(n, n),
        )
        # n = adj.size(0)
        device = edge_index.device
        self.initialized = True
        sp_eye = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=[1.0] * n,
            size=(n, n),
            dtype=torch.float,
        ).to(device)
        # initialize A1, A2
        a1 = self._indicator(adj - sp_eye)
        a2 = self._indicator(self._spspmm(adj, adj) - adj - sp_eye)
        # norm A1 A2
        self.a1 = self._adj_norm(a1)
        self.a2 = self._adj_norm(a2)

    # def forward(self, adj: torch.sparse.Tensor, x: FloatTensor) -> FloatTensor:
    def forward(self, data):
        if not self.initialized:
            edge_index = data.graph["edge_index"]
            num_nodes = data.graph["num_nodes"]
            self._prepare_prop(edge_index, num_nodes)
        # H2GCN propagation
        x = data.graph["node_feat"]

        if self.relu:
            rs = [F.relu(torch.mm(x, self.w_embed))]
        else:
            rs = [torch.mm(x, self.w_embed)]
        for i in range(self.k):
            r_last = rs[-1]
            r1 = torch.spmm(self.a1, r_last)
            r2 = torch.spmm(self.a2, r_last)
            rs.append(torch.cat([r1, r2], dim=1))
        r_final = torch.cat(rs, dim=1)
        r_final = F.dropout(r_final, self.dropout, training=self.training)
        return torch.softmax(torch.mm(r_final, self.w_classify), dim=1)


class SIGN(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        hops,
        num_layers,
        dropout,
        use_bn=False,
    ):
        super().__init__()
        # print(f'in_channel:{in_channels}')
        self.lins = torch.nn.ModuleList()
        self.lins.append(
            torch.nn.Linear(in_channels * (hops + 1), hidden_channels)
        )
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.num_layers = num_layers
        self.hops = hops
        self.use_bn = use_bn

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x = data.graph["node_feat"]
        edge_index = data.graph["edge_index"]
        N = x.shape[0]

        row, col = edge_index
        d = degree(col, N).float()
        d_norm_in = (1.0 / d[col]).sqrt()
        d_norm_out = (1.0 / d[row]).sqrt()
        value = torch.ones_like(row) * d_norm_in * d_norm_out
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))

        embedding = [x]
        for _ in range(self.hops):
            x = torch_sparse.matmul(adj, x)
            embedding.append(x)

        x = torch.cat(embedding, dim=1)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
