from ast import Tuple
import os
import re
from typing import Callable, Iterator
import lightning as L
from torch import nn, optim
import torch

from deepreaction.utils.standardizer import Standardizer


# TODO: Move supervised training and evaluation into subclass and generalize the class
class SupervisedRoutine(L.LightningModule):
    """"""
    def __init__(
            self, 
            model: nn.Module, 
            loss: Callable, 
            optimizer_factory: Callable[[Iterator[nn.Parameter]], optim.Optimizer],
            lr_scheduler: optim.lr_scheduler.LRScheduler = None,
            metrics=None,
            pretrained_path: str = None,
            resume_training: bool = False,
            standardizer_path=None
        ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer: optim.Optimizer = optimizer_factory(params=self.model.parameters())
        self.lr_scheduler = lr_scheduler
        self.metrics = metrics if metrics is not None else {}
        self.pretrained_path = pretrained_path
        self.resume_training = resume_training
        self.standardizer_path = standardizer_path
        # TODO: Instantiate or pass standardizer

    def setup(self, stage=None):
        """"""
        if self.pretrained_path:
            self._load_pretrained(self.pretrained_path, self.resume_training)
        if self.standardizer_path:
            self.standardizer = self._load_standardizer(self.standardizer_path)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # TODO: Use standardizer to normalize predictions if needed
        return self.model(input)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self._step(batch, stage="train", track_metrics=False)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        self._step(batch, stage="val", track_metrics=True)
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        self._step(batch, stage="test", track_metrics=True)
        
    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler,
        }
    
    def _load_pretrained(self, path: str, resume_training: bool = False):
        if not os.path.exists(path):
            raise ValueError(f"Pretrained path does not exist: {path}")

        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        # PyTorch deserializes checkpoint on CPU and then moves them to the device-type
        # it saved on (e.g. 'cuda') by default. This can lead to surged in memory usage
        # if the model is large. To avoid this, we load the checkpoint on CPU and move 
        # the model to the correct device later.
        # We also use `weights_only=True` to load only the model weights, out of securtiy
        # reasons.

        # Validate the checkpoint dictionary
        keys = ['model_state_dict']
        if resume_training:
            keys += ['optimizer_state_dict']
            if self.lr_scheduler:
                keys.append('lr_scheduler_state_dict')
        self._validate_checkpoint_dict(checkpoint, keys)

        # Load state dictionaries
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if resume_training:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.lr_scheduler:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

        self.log("pretrained_model_loaded", True, prog_bar=True)

    def _validate_checkpoint_dict(self, checkpoint: dict, keys: list):
        """
        Validate that the checkpoint dictionary contains the required keys.
        """
        missing_keys = [key for key in keys if key not in checkpoint]
        if missing_keys:
            raise ValueError(f"Checkpoint is missing required keys: {missing_keys}")
    
    def _load_standardizer(self, path: str):
        if not os.path.exists(path):
            raise ValueError(f"Standardizer path does not exist: {path}")
        params = torch.load(path)
        standardizer = Standardizer(mean=params['mean'], std=params['std'])
        self.log("standardizer_loaded", True, prog_bar=True)
        return standardizer
        

    def _step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str, track_metrics: bool=False) -> torch.Tensor:
        input, target = batch
        output = self(input)
        loss = self.loss(output, target)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        if track_metrics and self.metrics:
            for name, metric in self.metrics.items():
                value = metric(output, target)
                self.log(f"{stage}_{name}", value, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    