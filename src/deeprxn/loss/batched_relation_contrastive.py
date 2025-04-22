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
        reactant_batch_indices = kwargs["reactant_batch_indices"]
        product_batch_indices = kwargs["product_batch_indices"]

        mse_loss = self.mse(preds, y)
        contrastive_loss_total = 0.0

        unique_batches = torch.unique(reactant_batch_indices)
        for batch_id in unique_batches:
            r_mask = reactant_batch_indices == batch_id
            p_mask = product_batch_indices == batch_id

            if r_mask.sum() == 0 or p_mask.sum() == 0:
                assert False, f"Batch {batch_id} has no reactants or products."

            r_feat = reactant_features[r_mask]
            p_feat = product_features[p_mask]

            r_norm = F.normalize(r_feat, p=2, dim=1)
            p_norm = F.normalize(p_feat, p=2, dim=1)

            sim_matrix = torch.matmul(r_norm, p_norm.transpose(0, 1)) / self.temperature

            # assumes that the correct mapping is along the diagonal, so:
            labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
            contrastive_loss_total += F.cross_entropy(sim_matrix, labels)

        if unique_batches.numel() > 0:
            contrastive_loss = contrastive_loss_total / unique_batches.numel()
        else:
            contrastive_loss = 0.0

        total_loss = mse_loss + self.weight * contrastive_loss
        return total_loss

