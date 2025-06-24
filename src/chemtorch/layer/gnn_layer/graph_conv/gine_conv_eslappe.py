import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.nn import MessagePassing


class GINEConvESLapPE(MessagePassing):
    """GINEConv Layer with EquivStableLapPE implementation.

    Modified torch_geometric.nn.conv.GINEConv layer to perform message scaling
    according to equiv. stable PEG-layer with Laplacian Eigenmap (LapPE):
        ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        eps=0.0,
        train_eps=False,
        edge_dim=None,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super(GINEConvESLapPE, self).__init__(**kwargs)
        self.nn = nn.Sequential(
            Linear_pyg(in_channels, out_channels),
            nn.ReLU(),
            Linear_pyg(out_channels, out_channels),
        )
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))
        if edge_dim is not None:
            if hasattr(self.nn[0], "in_features"):
                in_channels = self.nn[0].in_features
            else:
                in_channels = self.nn[0].in_channels
            self.lin = pyg_nn.Linear(edge_dim, in_channels)
        else:
            self.lin = None

        if hasattr(self.nn[0], "in_features"):
            out_dim = self.nn[0].out_features
        else:
            out_dim = self.nn[0].out_channels

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        self.mlp_r_ij = torch.nn.Sequential(
            torch.nn.Linear(1, out_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(out_dim, 1),
            torch.nn.Sigmoid(),
        )

        self.reset_parameters()

    def reset_parameters(self):
        pyg_nn.inits.reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()
        pyg_nn.inits.reset(self.mlp_r_ij)

    def forward(self, batch):
        # if isinstance(x, Tensor):
        #     x: OptPairTensor = (x, x)

        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        pe_LapPE = batch.pe_EquivStableLapPE

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, PE=pe_LapPE, size=None
        )

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        batch.x = self.nn(out)

        return batch

    def message(self, x_j, edge_attr, PE_i, PE_j):
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError(
                "Node and edge feature dimensionalities do not "
                "match. Consider setting the 'edge_dim' "
                "attribute of 'GINEConv'"
            )

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        r_ij = ((PE_i - PE_j) ** 2).sum(dim=-1, keepdim=True)
        r_ij = self.mlp_r_ij(r_ij)  # the MLP is 1 dim --> hidden_dim --> 1 dim

        return ((x_j + edge_attr).relu()) * r_ij

    def __repr__(self):
        return f"{self.__class__.__name__}(nn={self.nn})"
