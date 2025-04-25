import torch


class RelationLoss(torch.nn.Module):
    def __init__(self, weight, atom_reduction, reduction="sum"):
        super(RelationLoss, self).__init__()
        self.mse = torch.nn.MSELoss(reduction=reduction)
        self.weight = weight
        self.atom_reduction = atom_reduction

    def forward(self, preds, y, **kwargs):
        reactant_features = kwargs["reactant_features"]
        product_features = kwargs["product_features"]
        reactant_batch_indices = kwargs["reactant_batch_indices"]
        product_batch_indices = kwargs["product_batch_indices"]

        mse_loss = self.mse(preds, y)
        atom_loss_total = 0.0

        unique_batches = torch.unique(reactant_batch_indices)
        for batch_id in unique_batches:
            r_mask = reactant_batch_indices == batch_id
            p_mask = product_batch_indices == batch_id

            if r_mask.sum() == 0 or p_mask.sum() == 0:
                assert False, f"Batch {batch_id} has no reactants or products."

            r_feat = reactant_features[r_mask]
            p_feat = product_features[p_mask]

            dists = torch.cdist(r_feat, p_feat, p=2)
            pos_dists = torch.diag(dists)
            if self.atom_reduction == "mean":
                pos_dists = pos_dists.mean()
            elif self.atom_reduction == "sum":
                pos_dists = pos_dists.sum()
            else:
                raise ValueError(f"Unknown atom_reduction: {self.atom_reduction}")
            
            atom_loss_total += pos_dists

        total_loss = mse_loss + self.weight * atom_loss_total
        return total_loss

