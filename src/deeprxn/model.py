import os

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_scatter import scatter_add, scatter_mean

from deeprxn.data import AtomOriginType


def save_model(model, optimizer, epoch, best_val_loss, model_path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        },
        model_path,
    )


def load_model(model, optimizer, model_path):
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["best_val_loss"]

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return model, optimizer, epoch, best_val_loss
    else:
        return model, optimizer, 0, float("inf")


class GNN(nn.Module):
    # TODO: Add Docstring
    # TODO add arguments for depth, hidden_size, dropout, number of layers for fnn (currently hard-coded)
    def __init__(
        self,
        num_node_features,
        num_edge_features,
        depth,
        hidden_size,
        dropout,
        pool_type="global",
        separate_nn=False,
        pool_real_only=False,
    ):
        super(GNN, self).__init__()
        self.depth = depth
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.pool_type = pool_type
        self.separate_nn = separate_nn
        self.pool_real_only = pool_real_only

        self.edge_init = nn.Linear(
            num_node_features + num_edge_features, self.hidden_size
        )
        self.convs = torch.nn.ModuleList()
        for _ in range(self.depth):
            self.convs.append(DMPNNConv(self.hidden_size, separate_nn))
        self.edge_to_node = nn.Linear(
            num_node_features + self.hidden_size, self.hidden_size
        )
        self.pool = (
            global_add_pool  # TODO: add option for other pooling methods
        )

        valid_pool_types = ["global", "reactants", "products", "dummy"]
        if pool_type not in valid_pool_types:
            raise ValueError(
                f"Invalid pool_type. Choose from {', '.join(valid_pool_types)}"
            )
        self.pool_type = pool_type

        #        self.ffn = nn.Linear(self.hidden_size, 1)
        layers = [
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1),
        ]
        self.ffn = nn.Sequential(*layers)

    def forward(self, data):
        # TODO: add docstring
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        atom_origin_type = data.atom_origin_type
        is_real_bond = (
            data.is_real_bond if hasattr(data, "is_real_bond") else None
        )

        # initial edge features
        row, col = edge_index
        h_0 = F.relu(self.edge_init(torch.cat([x[row], edge_attr], dim=1)))
        h = h_0

        # convolutions
        for l in range(self.depth):
            _, h = self.convs[l](edge_index, h, is_real_bond)
            h += h_0
            h = F.dropout(F.relu(h), self.dropout, training=self.training)

        # dmpnn edge -> node aggregation
        s, _ = self.convs[l](
            edge_index, h, is_real_bond
        )  # only use for summing
        q = torch.cat([x, s], dim=1)
        h = F.relu(self.edge_to_node(q))

        if self.pool_type == "global":
            if self.pool_real_only:
                pooled = self.pool_real_bonds(
                    h, batch, edge_index, is_real_bond
                )
            else:
                pooled = self.pool(h, batch)
        elif self.pool_type == "reactants":
            if self.pool_real_only:
                pooled = self.pool_real_bonds_reactants(
                    h, batch, edge_index, is_real_bond, atom_origin_type
                )
            else:
                pooled = self.pool_reactants(h, batch, atom_origin_type)
        elif self.pool_type == "products":
            if self.pool_real_only:
                pooled = self.pool_real_bonds_products(
                    h, batch, edge_index, is_real_bond, atom_origin_type
                )
            else:
                pooled = self.pool_products(h, batch, atom_origin_type)
        elif self.pool_type == "dummy":
            pooled = self.pool_dummy(h, batch, atom_origin_type)
        else:
            raise ValueError(
                "Invalid pool_type. Choose 'global', 'reactants', 'products', or 'dummy'."
            )

        assert (
            pooled.dim() == 2
        ), f"Expected 2D tensor, got {pooled.dim()}D tensor from {self.pool_type} pooling"

        output = self.ffn(pooled).squeeze(-1)
        return output

    def pool_reactants(self, h, batch, atom_types):
        return self.pool(
            h[atom_types == AtomOriginType.REACTANT],
            batch[atom_types == AtomOriginType.REACTANT],
        )

    def pool_products(self, h, batch, atom_types):
        return self.pool(
            h[atom_types == AtomOriginType.PRODUCT],
            batch[atom_types == AtomOriginType.PRODUCT],
        )

    def pool_dummy(self, h, batch, atom_types):
        dummy_mask = atom_types == AtomOriginType.DUMMY
        if not dummy_mask.any():
            raise ValueError("No dummy nodes found in the graph")

        dummy_h = h[dummy_mask]
        dummy_batch = batch[dummy_mask]

        pooled = scatter_add(
            dummy_h, dummy_batch, dim=0
        )  # TODO: add option for mean pooling

        return pooled

    def pool_real_bonds_reactants(
        self, h, batch, edge_index, is_real_bond, atom_types
    ):
        row, col = edge_index[:, is_real_bond]
        mask = atom_types[row] == AtomOriginType.REACTANT
        row, col = row[mask], col[mask]
        edge_batch = batch[row]
        pooled = scatter_add(h[row], edge_batch, dim=0)
        return pooled

    def pool_real_bonds_products(
        self, h, batch, edge_index, is_real_bond, atom_types
    ):
        row, col = edge_index[:, is_real_bond]
        mask = atom_types[row] == AtomOriginType.PRODUCT
        row, col = row[mask], col[mask]
        edge_batch = batch[row]
        pooled = scatter_add(h[row], edge_batch, dim=0)
        return pooled


class DMPNNConv(MessagePassing):
    # TODO: add docstring
    def __init__(self, hidden_size, separate_nn=False):
        super(DMPNNConv, self).__init__(aggr="add")
        self.lin_real = nn.Linear(hidden_size, hidden_size)
        self.lin_artificial = (
            nn.Linear(hidden_size, hidden_size)
            if separate_nn
            else self.lin_real
        )
        self.separate_nn = separate_nn

    def forward(self, edge_index, edge_attr, is_real_bond=None):
        # TODO: add docstring
        row, col = edge_index
        a_message = self.propagate(edge_index, x=None, edge_attr=edge_attr)

        try:
            rev_message = torch.flip(
                edge_attr.view(edge_attr.size(0) // 2, 2, -1), dims=[1]
            ).view(edge_attr.size(0), -1)
        except:
            rev_message = torch.zeros_like(edge_attr)

        if self.separate_nn and is_real_bond is not None:
            out = torch.where(
                is_real_bond.unsqueeze(1),
                self.lin_real(a_message[row] - rev_message),
                self.lin_artificial(a_message[row] - rev_message),
            )
        else:
            out = self.lin_real(a_message[row] - rev_message)

        return a_message, out

    def message(self, edge_attr):
        # TODO: add docstring
        return edge_attr
