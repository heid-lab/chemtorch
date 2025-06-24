import math
import os
import time

import hydra
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score
from torch import nn

from chemtorch.utils.misc import check_early_stopping, load_model, save_model


def predict(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for (
            data_x,
            _,
        ) in loader:
            data_x = data_x.to(device)

            out = model(data_x)

            predicted_batch_indices = torch.argmax(out, dim=1)

            preds.extend(predicted_batch_indices.cpu().tolist())
    return preds


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data_x, data_y in loader:
            data_x = data_x.to(device)
            data_y = data_y.to(device)

            out = model(data_x)

            loss = loss_fn(out, data_y.long())

            total_loss += loss.item() * data_y.size(0)

            predicted_batch_indices = torch.argmax(out, dim=1)

            all_preds.extend(predicted_batch_indices.cpu().tolist())
            all_labels.extend(data_y.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def train_epoch(
    model,
    train_loader,
    optimizer,
    loss_fn,
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
        loss = loss_fn(out, data_y)
        loss.backward()

        if clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip_grad_norm_value
            )
        optimizer.step()
        loss_all += loss.item() * data_y.size(0)

    epoch_time = time.time() - start_time
    return loss_all / len(train_loader.dataset), epoch_time


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
    loss,
    optimizer,
    lr_scheduler,
    use_wandb=False,
):

    if use_wandb:
        wandb.run.summary["status"] = "training"

    requires_metric = getattr(lr_scheduler, "requires_metric", False)

    optimizer_partial = hydra.utils.instantiate(optimizer)
    optimizer = optimizer_partial(params=model.parameters())
    scheduler_partial = hydra.utils.instantiate(lr_scheduler)
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
            device=device,
            clip_grad_norm=clip_grad_norm,
            clip_grad_norm_value=clip_grad_norm_value,
        )
        val_bce_loss, val_accuracy = evaluate(
            model, val_loader, loss_fn, device
        )

        try:
            if requires_metric:
                scheduler.step(val_bce_loss)
            else:
                scheduler.step()
        except TypeError as e:
            raise TypeError(
                f"Scheduler step failed. Check if requires_metric is properly configured: {e}"
            )

        early_stop_counter, should_stop = check_early_stopping(
            val_bce_loss,
            best_val_loss,
            early_stop_counter,
            patience,
            min_delta,
        )

        if val_bce_loss < best_val_loss:
            best_val_loss = val_bce_loss
            if save_model_parameters:
                save_model(model, optimizer, epoch, best_val_loss, model_path)

        print(
            f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_bce_loss:.4f}, Val Acc: {val_accuracy:.4f}, Time: {epoch_time:.2f}"
        )

        current_lr = optimizer.param_groups[0]["lr"]

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_bce_loss": train_loss,  # Changed from train_rmse
                    "val_bce_loss": val_bce_loss,  # Changed from val_rmse
                    "val_accuracy": val_accuracy,  # Added validation accuracy
                    "best_val_bce_loss": best_val_loss,  # Changed from best_val_rmse
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
    test_bce_loss, test_accuracy = evaluate(
        model, test_loader, loss_fn, device
    )

    print(f"Test BCE Loss: {test_bce_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    if use_wandb:
        wandb.log(
            {"test_bce_loss": test_bce_loss, "test_accuracy": test_accuracy}
        )
        wandb.run.summary["status"] = "completed"
