import torch
import torch.nn.functional as F

import wandb


class AtomMapContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.1, reduction="mean"):
        super(AtomMapContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction=reduction)
    
    def forward(self, **kwargs):
        reactant_features = kwargs["reactant_features"]
        product_features = kwargs["product_features"]
        reactant_batch_indices = kwargs["reactant_batch_indices"]
        product_batch_indices = kwargs["product_batch_indices"]

        contrastive_loss_total = 0.0
        num_valid_batches = 0

        unique_batches = torch.unique(reactant_batch_indices)
        for batch_id in unique_batches:
            r_mask = reactant_batch_indices == batch_id
            p_mask = product_batch_indices == batch_id

            # Skip if reaction has 0 reactants or 0 products (e.g., degenerate cases)
            # Or if reactant/product counts don't match (implies mapping isn't 1-to-1 diagonal)
            if r_mask.sum() == 0 or p_mask.sum() == 0 or r_mask.sum() != p_mask.sum():
                print(f"Warning: Skipping batch {batch_id} due to mismatched/zero atom counts (R: {r_mask.sum()}, P: {p_mask.sum()}).")
                continue

            r_feat = reactant_features[r_mask]
            p_feat = product_features[p_mask]

            if r_feat.shape[0] == 0:
                continue

            r_norm = F.normalize(r_feat, p=2, dim=1)
            p_norm = F.normalize(p_feat, p=2, dim=1)

            sim_matrix = torch.matmul(r_norm, p_norm.transpose(0, 1)) / self.temperature

            labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)

            contrastive_loss_total += self.ce_loss(sim_matrix, labels)
            num_valid_batches += 1

        if num_valid_batches > 0:
            contrastive_loss = contrastive_loss_total / num_valid_batches
        else:
            assert False, "No valid batches for contrastive loss calculation. Check your data."

        return contrastive_loss

