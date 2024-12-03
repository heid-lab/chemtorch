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
from deeprxn.predict import predict
from deeprxn.utils import (
    load_model,
    load_standardizer,
    save_model,
    save_standardizer,
)


def train_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    loss,
    stdzer,
    device,
    clip_grad_norm,
    clip_grad_norm_value,
):
    start_time = time.time()
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data)
        result = loss(out, stdzer(data.y))
        result.backward()

        if clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip_grad_norm_value
            )
        optimizer.step()
        loss_all += loss(stdzer(out, rev=True), data.y)

    scheduler.step()

    epoch_time = time.time() - start_time
    return math.sqrt(loss_all / len(train_loader.dataset)), epoch_time


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


def train(train_loader, val_loader, test_loader, cfg):
    # TODO add docstring

    device = torch.device(cfg.device)

    mean = np.mean(train_loader.dataset.labels)
    std = np.std(train_loader.dataset.labels)
    stdzer = Standardizer(mean, std)

    model = hydra.utils.instantiate(cfg.model)
    model = model.to(device)

    optimizer_partial = hydra.utils.instantiate(cfg.optimizer)
    optimizer = optimizer_partial(params=model.parameters())
    scheduler_partial = hydra.utils.instantiate(cfg.scheduler)
    scheduler = scheduler_partial(optimizer)

    loss = nn.MSELoss(reduction="sum")
    print(model)

    if cfg.wandb:
        wandb.watch(model, log="all")

    # TODO: add support for continuing training
    # model, optimizer, start_epoch, best_val_loss = load_model(
    #     model, optimizer, cfg.model_path
    # )
    start_epoch = 0
    best_val_loss = float("inf")

    early_stop_counter = 0
    for epoch in range(start_epoch, cfg.epochs):
        train_loss, epoch_time = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss=loss,
            stdzer=stdzer,
            device=device,
            clip_grad_norm=cfg.clip_grad_norm_value,
            clip_grad_norm_value=cfg.clip_grad_norm_value,
        )
        val_preds = predict(model, val_loader, stdzer, device)
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
                save_standardizer(mean, std, cfg.model_path)

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

    # final evaluation
    # TODO: look into keeping track of best model state, now we use last model state
    if cfg.save_model:
        model, _, _, _ = load_model(model, optimizer, cfg.model_path)
    test_preds = predict(model, test_loader, stdzer, device)
    test_rmse = root_mean_squared_error(test_preds, test_loader.dataset.labels)
    test_mae = mean_absolute_error(test_preds, test_loader.dataset.labels)
    print(f"Test RMSE: {test_rmse}")
    print(f"Test MAE: {test_mae}")

    if cfg.wandb:
        wandb.log({"test_rmse": test_rmse, "test_mae": test_mae})
