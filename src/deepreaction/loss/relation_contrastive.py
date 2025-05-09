import torch
import torch.nn.functional as F

import wandb


class RelationLoss(torch.nn.Module):
    def __init__(self, temperature=0.1, weight=0.1, reduction="sum"):
        super(RelationLoss, self).__init__()
        self.mse = torch.nn.MSELoss(reduction=reduction)
        self.temperature = temperature
        self.weight = weight
    
    def forward(self, preds, y, **kwargs):
        reactant_features = kwargs["reactant_features"]
        product_features = kwargs["product_features"]
        
        r_norm = F.normalize(reactant_features, p=2, dim=1)
        p_norm = F.normalize(product_features, p=2, dim=1)
        
        sim_matrix = torch.matmul(r_norm, p_norm.transpose(0, 1)) / self.temperature
        
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        
        contrastive_loss = F.cross_entropy(sim_matrix, labels)
        
        mse_loss = self.mse(preds, y)
        
        total_loss = mse_loss + self.weight * contrastive_loss

        return total_loss
