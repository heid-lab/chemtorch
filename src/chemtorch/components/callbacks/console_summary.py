import time
import torch
import lightning as L
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf, DictConfig


class ConsoleSummaryCallback(Callback):
    """
    A custom callback to print metrics and optionally the model summary and config.

    Ideal for getting immediate feedback during training, especially when the
    default progress bar is disabled.
    """

    def __init__(self, print_config: bool = True, print_summary: bool = True):
        """
        Args:
            print_config (bool): If True, prints the resolved Hydra/OmegaConf config
                at the start of training. Defaults to True.
            print_summary (bool): If True, prints the Lightning model summary
                at the start of training. Defaults to True.
        """
        super().__init__()
        self.print_config = print_config
        self.print_summary = print_summary
        self.epoch_start_time = 0

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Hook to optionally print the config and model summary at the start of training."""

        if self.print_config:
            if hasattr(pl_module, "config") and isinstance(
                pl_module.config, DictConfig
            ):
                pl_module.print(OmegaConf.to_yaml(pl_module.config))
                pl_module.print("-----------------------------")
            else:
                pl_module.print(
                    "[Callback Warning] `print_config=True` but no 'config' "
                    "attribute was found on the LightningModule."
                )

        if self.print_summary:
            pl_module.print(trainer.model)
            pl_module.print("---------------------\n")

    def on_train_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Hook to record the start time of a training epoch."""
        self.epoch_start_time = time.monotonic()

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Hook to print metrics at the end of a validation epoch."""
        if trainer.sanity_checking:
            return

        epoch_duration = time.monotonic() - self.epoch_start_time

        metrics = trainer.logged_metrics
        train_rmse = metrics.get("train_rmse_epoch", metrics.get("train_rmse", "N/A"))
        val_rmse = metrics.get("val_rmse_epoch", metrics.get("val_rmse", "N/A"))

        if isinstance(train_rmse, torch.Tensor):
            train_rmse = f"{train_rmse.item():.4f}"
        if isinstance(val_rmse, torch.Tensor):
            val_rmse = f"{val_rmse.item():.4f}"

        pl_module.print(
            f"Epoch {trainer.current_epoch}: "
            f"Train RMSE: {train_rmse} | "
            f"Val RMSE: {val_rmse} | "
            f"Time: {epoch_duration:.2f}s"
        )
