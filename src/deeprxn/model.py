import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool
import os

def save_model(model, optimizer, epoch, best_val_loss, model_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }, model_path)

def load_model(model, optimizer, model_path):
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return model, optimizer, epoch, best_val_loss
    else:
        return model, optimizer, 0, float('inf')

class GNN(nn.Module):
    #TODO: Add Docstring
    #TODO add arguments for depth, hidden_size, dropout, number of layers for fnn (currently hard-coded)
    def __init__(self, num_node_features, num_edge_features, pool_type="global", bidirectional=True):
        super(GNN, self).__init__()

        self.depth = 3
        self.hidden_size = 300
        self.dropout = 0.02

        self.pool_type = pool_type

        self.edge_init = nn.Linear(num_node_features + num_edge_features, self.hidden_size)
        self.convs = torch.nn.ModuleList()
        for _ in range(self.depth):
            self.convs.append(DMPNNConv(self.hidden_size, bidirectional))
        self.edge_to_node = nn.Linear(num_node_features + self.hidden_size, self.hidden_size)
        self.pool = global_add_pool
        
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
        #TODO: add docstring
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        atom_is_reactant = data.atom_is_reactant

        # initial edge features
        row, col = edge_index
        h_0 = F.relu(self.edge_init(torch.cat([x[row], edge_attr], dim=1)))
        h = h_0

        # convolutions
        for l in range(self.depth):
            _, h = self.convs[l](edge_index, h)
            h += h_0
            h = F.dropout(F.relu(h), self.dropout, training=self.training)

        # dmpnn edge -> node aggregation
        s, _ = self.convs[l](edge_index, h) #only use for summing
        q  = torch.cat([x,s], dim=1)
        h = F.relu(self.edge_to_node(q))

        if self.pool_type == "global":
            pooled = self.pool(h, batch)
        elif self.pool_type == "reactants":
            pooled = self.pool_reactants(h, batch, atom_is_reactant)
        elif self.pool_type == "products":
            pooled = self.pool_products(h, batch, atom_is_reactant)
        else:
            raise ValueError("Invalid pool_type. Choose 'global', 'reactants', or 'products'.")

        return self.ffn(pooled).squeeze(-1)
    
    def pool_reactants(self, h, batch, atom_is_reactant):
        return self.pool(h[atom_is_reactant], batch[atom_is_reactant])

    def pool_products(self, h, batch, atom_is_reactant):
        return self.pool(h[~atom_is_reactant], batch[~atom_is_reactant])

class DMPNNConv(MessagePassing):
    #TODO: add docstring 
    def __init__(self, hidden_size, bidirectional=True):
        super(DMPNNConv, self).__init__(aggr='add')
        self.lin = nn.Linear(hidden_size, hidden_size)
        self.bidirectional = bidirectional

    def forward(self, edge_index, edge_attr):
        #TODO: add docstring
        row, col = edge_index
        a_message = self.propagate(edge_index, x=None, edge_attr=edge_attr)

        if self.bidirectional:
            rev_message = torch.flip(edge_attr.view(edge_attr.size(0) // 2, 2, -1), dims=[1]).view(edge_attr.size(0), -1)
        else:
            rev_message = torch.zeros_like(edge_attr)

        return a_message, self.lin(a_message[row] - rev_message)

    def message(self, edge_attr):
        #TODO: add docstring
        return edge_attr
