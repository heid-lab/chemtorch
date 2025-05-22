import math
import os
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from torch import nn

import wandb
from deepreaction.misc import (
    check_early_stopping,
    load_model,
    load_standardizer,
    save_model,
    save_standardizer,
)
from deepreaction.standardizer import Standardizer

# torch.autograd.set_detect_anomaly(True)


def predict(model, loader, stdzer, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for data_x, data_y in loader:
            data_x = data_x.to(device)
            data_y = data_y.to(device)
            out = model(data_x)         # shape: (batch_size, 1)
            out = out.squeeze(-1)       # shape: (batch_size)
            pred = stdzer(out, rev=True)
            preds.extend(pred.cpu().detach().tolist())
    return preds


def train_epoch(
    model,
    train_loader,
    optimizer,
    loss_fn,
    stdzer,
    device,
    clip_grad_norm,
    clip_grad_norm_value,
):
    start_time = time.time()
    model.train()
    loss_all = 0

    for data_x, data_y in train_loader:
        data_x = data_x.to(device)
        data_y = data_y.to(device)
        optimizer.zero_grad()

        out = model(data_x)
        out = out.squeeze(-1)
        result = loss_fn(out, stdzer(data_y))
        result.backward()

        if clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip_grad_norm_value
            )
        optimizer.step()
        loss_all += loss_fn(stdzer(out, rev=True), data_y)

    epoch_time = time.time() - start_time
    return math.sqrt(loss_all / len(train_loader.dataset)), epoch_time


def train(
    train_loader,
    val_loader,
    test_loader,
    model,
    device,
    epochs,
    clip_grad_norm,
    clip_grad_norm_value,
    patience,
    min_delta,
    save_model_parameters,
    model_path,
    loss: DictConfig,
    optimizer: DictConfig,
    scheduler: DictConfig,
    use_wandb=False,
):

    train_labels = train_loader.dataset.get_labels()
    mean = np.mean(train_labels)
    std = np.std(train_labels)
    stdzer = Standardizer(mean, std)

    if use_wandb:
        wandb.run.summary["status"] = "training"

    requires_metric = getattr(scheduler, "requires_metric", False)

    optimizer_partial = hydra.utils.instantiate(optimizer)
    optimizer = optimizer_partial(params=model.parameters())
    scheduler_partial = hydra.utils.instantiate(scheduler.scheduler)
    scheduler = scheduler_partial(optimizer)

    loss_fn = hydra.utils.instantiate(loss)
    print(model)

    if use_wandb:
        wandb.watch(model, log="all")

    start_epoch = 0
    best_val_loss = float("inf")

    early_stop_counter = 0
    for epoch in range(start_epoch, epochs):
        train_loss, epoch_time = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            stdzer=stdzer,
            device=device,
            clip_grad_norm=clip_grad_norm,
            clip_grad_norm_value=clip_grad_norm_value,
        )
        val_preds = predict(model, val_loader, stdzer, device)
        val_loss = root_mean_squared_error(
            val_preds, val_loader.dataset.get_labels()
        )

        try:
            if requires_metric:
                scheduler.step(val_loss)
            else:
                scheduler.step()
        except TypeError as e:
            raise TypeError(
                f"Scheduler step failed. Check if requires_metric is properly configured: {e}"
            )

        early_stop_counter, should_stop = check_early_stopping(
            val_loss,
            best_val_loss,
            early_stop_counter,
            patience,
            min_delta,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_model_parameters:
                save_model(model, optimizer, epoch, best_val_loss, model_path)
                save_standardizer(mean, std, model_path)

        print(
            f"Epoch {epoch}, Train RMSE: {train_loss}, Val RMSE: {val_loss}, Time: {epoch_time:.2f}"
        )

        current_lr = optimizer.param_groups[0]["lr"]

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_rmse": train_loss,
                    "val_rmse": val_loss,
                    "best_val_rmse": best_val_loss,
                    "learning_rate": current_lr,
                }
            )

        if should_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    # final evaluation
    # TODO: look into keeping track of best model state, now we use last model state
    if save_model_parameters:
        model, _, _, _ = load_model(model, optimizer, model_path)
    test_preds = predict(model, test_loader, stdzer, device)
    test_labels = test_loader.dataset.get_labels()
    test_rmse = root_mean_squared_error(test_preds, test_labels)
    test_mae = mean_absolute_error(test_preds, test_labels)
    print(f"Test RMSE: {test_rmse}")
    print(f"Test MAE: {test_mae}")

    if use_wandb:
        wandb.log({"test_rmse": test_rmse, "test_mae": test_mae})
        wandb.run.summary["status"] = "completed"
