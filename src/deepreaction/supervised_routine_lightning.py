from ast import Tuple
from typing import Callable
import lightning as L
from torch import nn, optim
import torch


class SupervisedRoutine(L.LightningModule):
    """"""
    def __init__(
            self, 
            model: nn.Module, 
            loss: Callable, 
            optimizer_factory: optim.Optimizer,
            lr_scheduler: optim.lr_scheduler.LRScheduler = None,
            metrics=None
        ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer_factory
        self.lr_scheduler = lr_scheduler
        self.metrics = metrics if metrics is not None else {}

    def setup():
        """"""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.model(input)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        input, target = batch
        output = self(input)
        loss = self.loss(output, target)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        input, target = batch
        output = self(input)
        loss = self.loss(output, target)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Log metrics if available
        for name, metric in self.metrics.items():
            value = metric(output, target)
            self.log(f"val_{name}", value, on_step=False, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        input, target = batch
        output = self(input)
        loss = self.loss(output, target)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Log metrics if available
        for name, metric in self.metrics.items():
            value = metric(output, target)
            self.log(f"test_{name}", value, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler,  # Customoize by return a lr_scheduler_config dict
        }
    