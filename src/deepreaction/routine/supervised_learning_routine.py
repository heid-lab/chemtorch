from ast import Tuple
import os
from typing_extensions import override
from typing import Any, Callable, Iterator
from torch import nn, optim
import torch
from deepreaction.routine.routine_base import RoutineBase
from deepreaction.utils.standardizer import Standardizer


class SupervisedLearningRoutine(RoutineBase):
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
        super().__init__(
            model=model, 
            loss=loss, 
            optimizer_factory=optimizer_factory, 
            lr_scheduler=lr_scheduler, 
            metrics=metrics, 
            pretrained_path=pretrained_path, 
            resume_training=resume_training
        )
        self.standardizer_path = standardizer_path
    
    @override
    def setup(self, stage=None):
        super().setup(stage)
        # Load standardizer from file if provided
        if self.standardizer_path:
            self.standardizer = torch.load(self.standardizer_path)

    @override
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.model(inputs)
        if hasattr(self, 'standardizer'):
            outputs = self.standardizer(outputs, rev=True)
        return outputs

    @override
    def step(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor], 
        stage: str, 
        track_metrics: bool = False
    ) -> torch.Tensor:
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.loss(outputs, targets)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        if track_metrics and self.metrics:
            for name, metric in self.metrics.items():
                value = metric(outputs, targets)
                self.log(f"{stage}_{name}", value, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def _load_standardizer(self, path: str):
        if not os.path.exists(path):
            raise ValueError(f"Standardizer path does not exist: {path}")
        params = torch.load(path)
        standardizer = Standardizer(mean=params['mean'], std=params['std'])
        self.log("standardizer_loaded", True, prog_bar=True)
        return standardizer