import math
import time

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from torch import nn

import wandb
from deeprxn.data import Standardizer
from deeprxn.model import GNN
from deeprxn.utils import load_model, save_model


def train_epoch(model, loader, optimizer, loss, stdzer, device):
    start_time = time.time()
    model.train()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data)
        result = loss(out, stdzer(data.y))
        result.backward()

        optimizer.step()
        loss_all += loss(stdzer(out, rev=True), data.y)

    epoch_time = time.time() - start_time
    return math.sqrt(loss_all / len(loader.dataset)), epoch_time


def check_early_stopping(
    current_loss, best_loss, counter, patience, min_delta
):
    if current_loss < best_loss - min_delta:
        return 0, False
    else:
        counter += 1
        if counter >= patience:
            return counter, True
        return counter, False


def pred(model, loader, loss, stdzer, device):
    # TODO: add docstring
    model.eval()

    preds, ys = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            pred = stdzer(out, rev=True)
            preds.extend(pred.cpu().detach().tolist())

    return preds


def train(train_loader, val_loader, test_loader, cfg):
    # TODO add docstring

    cfg.model.num_node_features = train_loader.dataset.num_node_features
    cfg.model.num_edge_features = train_loader.dataset.num_edge_features

    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    resolved_cfg = OmegaConf.create(resolved_cfg)
    print(OmegaConf.to_yaml(resolved_cfg))

    device = torch.device(cfg.device)

    mean = np.mean(train_loader.dataset.labels)
    std = np.std(train_loader.dataset.labels)
    stdzer = Standardizer(mean, std)

    model = hydra.utils.instantiate(cfg.model)
    model = model.to(device)

    optimizer_params = {"lr": cfg.learning_rate}
    if hasattr(cfg, "weight_decay"):
        optimizer_params["weight_decay"] = cfg.weight_decay
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)

    loss = nn.MSELoss(reduction="sum")
    print(model)

    if cfg.wandb:
        wandb.watch(model, log="all")

    model, optimizer, start_epoch, best_val_loss = load_model(
        model, optimizer, cfg.model_path
    )

    early_stop_counter = 0
    for epoch in range(start_epoch, cfg.epochs):
        train_loss, epoch_time = train_epoch(
            model, train_loader, optimizer, loss, stdzer, device
        )
        val_preds = pred(model, val_loader, loss, stdzer, device)
        val_loss = root_mean_squared_error(
            val_preds, val_loader.dataset.labels
        )

        early_stop_counter, should_stop = check_early_stopping(
            val_loss,
            best_val_loss,
            early_stop_counter,
            cfg.patience,
            cfg.min_delta,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if cfg.save_model:
                save_model(
                    model, optimizer, epoch, best_val_loss, cfg.model_path
                )

        print(
            f"Epoch {epoch}, Train RMSE: {train_loss}, Val RMSE: {val_loss}, Time: {epoch_time:.2f}"
        )

        if cfg.wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_rmse": train_loss,
                    "val_rmse": val_loss,
                    "best_val_rmse": best_val_loss,
                }
            )

        if should_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load the best model for final evaluation
    model, _, _, _ = load_model(model, optimizer, cfg.model_path)
    test_preds = pred(model, test_loader, loss, stdzer, device)
    test_rmse = root_mean_squared_error(test_preds, test_loader.dataset.labels)
    test_mae = mean_absolute_error(test_preds, test_loader.dataset.labels)
    print(f"Test RMSE: {test_rmse}")
    print(f"Test MAE: {test_mae}")

    if cfg.wandb:
        wandb.log({"test_rmse": test_rmse, "test_mae": test_mae})


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


from deeprxn.SGFormer import GCN, SGFormer


def train2(train_loader, val_loader, test_loader, cfg):
    # TODO add docstring

    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    resolved_cfg = OmegaConf.create(resolved_cfg)
    print(OmegaConf.to_yaml(resolved_cfg))

    device = torch.device(cfg.device)

    mean = np.mean(train_loader.dataset.labels)
    std = np.std(train_loader.dataset.labels)
    stdzer = Standardizer(mean, std)

    gnn = GNN(**cfg.model).to(device)

    model = SGFormer(
        cfg.model.num_node_features,
        300,
        1,
        num_layers=1,
        alpha=0.5,
        dropout=0.2,
        num_heads=1,
        use_bn=True,
        use_residual=True,
        use_graph=True,
        use_weight=True,
        use_act=True,
        graph_weight=0.8,
        gnn=gnn,
        aggregate="add",
    ).to(device)

    optimizer = torch.optim.Adam(
        [
            {"params": model.params1, "weight_decay": 0.001},
            {"params": model.params2, "weight_decay": 5e-4},
        ],
        lr=0.01,
    )

    loss = nn.MSELoss(reduction="sum")
    print(model)

    if cfg.wandb:
        wandb.watch(model, log="all")

    model, optimizer, start_epoch, best_val_loss = load_model(
        model, optimizer, cfg.model_path
    )

    early_stop_counter = 0
    for epoch in range(start_epoch, cfg.epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, loss, stdzer, device
        )
        val_preds = pred(model, val_loader, loss, stdzer, device)
        val_loss = root_mean_squared_error(
            val_preds, val_loader.dataset.labels
        )

        early_stop_counter, should_stop = check_early_stopping(
            val_loss,
            best_val_loss,
            early_stop_counter,
            cfg.patience,
            cfg.min_delta,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if cfg.save_model:
                save_model(
                    model, optimizer, epoch, best_val_loss, cfg.model_path
                )

        print(f"Epoch {epoch}, Train RMSE: {train_loss}, Val RMSE: {val_loss}")

        if cfg.wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_rmse": train_loss,
                    "val_rmse": val_loss,
                    "best_val_rmse": best_val_loss,
                }
            )

        if should_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load the best model for final evaluation
    model, _, _, _ = load_model(model, optimizer, cfg.model_path)
    test_preds = pred(model, test_loader, loss, stdzer, device)
    test_rmse = root_mean_squared_error(test_preds, test_loader.dataset.labels)
    test_mae = mean_absolute_error(test_preds, test_loader.dataset.labels)
    print(f"Test RMSE: {test_rmse}")
    print(f"Test MAE: {test_mae}")

    if cfg.wandb:
        wandb.log({"test_rmse": test_rmse, "test_mae": test_mae})
