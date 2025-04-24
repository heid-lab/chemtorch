import torch


class MSELoss(torch.nn.Module):
    def __init__(self, reduction="sum"):
        super(MSELoss, self).__init__()
        self.mse = torch.nn.MSELoss(reduction=reduction)

    def forward(self, preds, y):
        return self.mse(preds, y)

