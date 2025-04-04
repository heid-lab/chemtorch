import hydra
import torch
from omegaconf import OmegaConf
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

import wandb
from deeprxn.data import Standardizer
from deeprxn.utils import load_model, load_standardizer


def predict(model, loader, stdzer, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            pred = stdzer(out, rev=True)
            preds.extend(pred.cpu().detach().tolist())
    return preds


def predict_model(test_loader, cfg):
    """Run prediction using a saved model."""
    device = torch.device(cfg.device)

    OmegaConf.update(
        cfg,
        "model.num_node_features",
        test_loader.dataset.num_node_features,
        merge=True,
    )
    OmegaConf.update(
        cfg,
        "model.num_edge_features",
        test_loader.dataset.num_edge_features,
        merge=True,
    )
    model = hydra.utils.instantiate(cfg.model)
    model = model.to(device)

    model, _, _, _ = load_model(model, None, cfg.model_path)

    mean, std = load_standardizer(cfg.model_path)
    if mean is None or std is None:
        raise ValueError("No standardizer found. Model must be trained first.")

    stdzer = Standardizer(mean, std)

    test_preds = predict(model, test_loader, stdzer, device)

    test_rmse = root_mean_squared_error(test_preds, test_loader.dataset.labels)
    test_mae = mean_absolute_error(test_preds, test_loader.dataset.labels)
    print(f"Test RMSE: {test_rmse}")
    print(f"Test MAE: {test_mae}")

    if cfg.wandb:
        wandb.log({"test_rmse": test_rmse, "test_mae": test_mae})

    return test_preds
