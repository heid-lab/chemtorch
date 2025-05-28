from abc import ABC, abstractmethod
from ast import Tuple
import os
from typing import Any, Callable, Iterator
import lightning as L
from torch import nn, optim
import torch



class RoutineBase(ABC, L.LightningModule):
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
        ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer: optim.Optimizer = optimizer_factory(params=self.model.parameters())
        self.lr_scheduler = lr_scheduler
        self.metrics = metrics if metrics is not None else {}
        self.pretrained_path = pretrained_path
        self.resume_training = resume_training

    @abstractmethod
    def step(self, batch: Any, stage: str, track_metrics: bool=False) -> torch.Tensor:
        """
        Perform a single step of training, validation, or testing.
        Subclasses should implement this method to define the specific behavior
        for each step in the routine.

        Args:
            batch (Any): The input batch of data.
            stage (str): The stage of the routine ('train', 'val', 'test').
            track_metrics (bool): Whether to track metrics for this step.

        Returns:
            torch.Tensor: The computed loss for the step.
        """
        pass

    def setup(self, stage=None):
        # Default implementation, override in subclass if needed and call super().setup(stage)
        if self.pretrained_path:
            self._load_pretrained(self.pretrained_path, self.resume_training)

    def forward(self, inputs: Any) -> Any:
        # Default implementation, override in subclass if needed
        return self.model(inputs)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.step(batch, stage="train", track_metrics=False)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        self.step(batch, stage="val", track_metrics=True)
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        self.step(batch, stage="test", track_metrics=True)
        
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