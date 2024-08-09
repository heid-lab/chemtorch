import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch import nn
import math
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from deeprxn.model import load_model, save_model
from deeprxn.data import Standardizer
from deeprxn.model import GNN

def train_epoch(model, loader, optimizer, loss, stdzer, device):
    #TODO: add docstring
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

    return math.sqrt(loss_all / len(loader.dataset))

def check_early_stopping(current_loss, best_loss, counter, patience, min_delta):
    if current_loss < best_loss - min_delta:
        return 0, False
    else:
        counter += 1
        if counter >= patience:
            return counter, True
        return counter, False

def pred(model, loader, loss, stdzer, device):
    #TODO: add docstring
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
    #TODO add docstring

    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    resolved_cfg = OmegaConf.create(resolved_cfg)
    print(OmegaConf.to_yaml(resolved_cfg))

    device = torch.device(cfg.device)

    mean = np.mean(train_loader.dataset.labels)
    std = np.std(train_loader.dataset.labels)
    stdzer = Standardizer(mean, std)

    bidirectional = cfg.transformation.connection_direction == "bidirectional"

    model = GNN(**cfg.model)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    loss = nn.MSELoss(reduction='sum')
    print(model)

    model, optimizer, start_epoch, best_val_loss = load_model(model, optimizer, cfg.model_path)
    model.to(device)

    early_stop_counter = 0
    for epoch in range(start_epoch, cfg.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss, stdzer, device)
        val_preds = pred(model, val_loader, loss, stdzer, device)
        val_loss = root_mean_squared_error(val_preds, val_loader.dataset.labels)
        
        early_stop_counter, should_stop = check_early_stopping(
            val_loss, best_val_loss, early_stop_counter, cfg.patience, cfg.min_delta
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if cfg.save_model:
                save_model(model, optimizer, epoch, best_val_loss, cfg.model_path)
          
        print(f"Epoch {epoch}, Train RMSE: {train_loss}, Val RMSE: {val_loss}")

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
