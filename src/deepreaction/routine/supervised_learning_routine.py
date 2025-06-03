from typing import Dict, Tuple, Callable, Iterator, Literal

import os
import torch
import lightning as L

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import Metric, MetricCollection


from deepreaction.utils.standardizer import Standardizer



class SupervisedLearningRoutine(L.LightningModule):
    """
    A routine for supervised learning tasks using PyTorch Lightning.
    """
    def __init__(
            self, 
            model: nn.Module, 
            loss: Callable, 
            optimizer: Callable[[Iterator[nn.Parameter]], Optimizer],
            lr_scheduler: Callable[[Optimizer], LRScheduler] = None,
            metrics: MetricCollection | Dict[str, MetricCollection] = None,
            standardizer_path: str = None,
            pretrained_path: str = None,
            resume_training: bool = False,
        ):
        """
        Initialize the SupervisedLearningRoutine.
        
        Args:
            model (nn.Module): The model to be trained.
            loss (Callable): The loss function to be used.
            optimizer (Callable): A factory function that takes in the models parameters 
                and returns an optimizer instance. For example, a partially instantiated 
                PyTorch optimizer.
            lr_scheduler (Callable, optional): A factory function that takes in the optimizer
                and returns a learning rate scheduler instance. Defaults to `None`, which means no scheduler is used.
            metrics (MetricCollection | Dict[str, MetricCollection], optional): A collection of metrics to be used for evaluation.
                It can be a single MetricCollection or a dictionary of metric collections with keys 'train', 
                'val', or 'test'. Defaults to `None`, which means no metrics are used.
            standardizer_path (str, optional): Path to a standardizer model. Defaults to `None`.
            pretrained_path (str, optional): Path to a pre-trained model checkpoint. Defaults to `None`.
            resume_training (bool, optional): Whether to resume training from a checkpoint. Defaults to `False`.
        
        Raises:
            TypeError: If `metrics` is not a MetricCollection or a dictionary of MetricCollections.
            ValueError: If `metrics` is a dictionary, but its keys are not 'train', 'val', or 'test',
                or if the keys are not unique.
        """
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer: Optimizer = optimizer(params=self.model.parameters())
        self.lr_scheduler: LRScheduler = lr_scheduler(self.optimizer) if lr_scheduler else None
        self.metrics = self._init_metrics(metrics) if metrics else None
        self.standardizer_path = standardizer_path
        self.pretrained_path = pretrained_path
        self.resume_training = resume_training

    ########## Lightning DataModule Methods ##############################################
    def setup(self, stage: str = None):
        if self.pretrained_path:
            self._load_pretrained(self.pretrained_path, self.resume_training)
        if self.standardizer_path:
            self.standardizer = torch.load(self.standardizer_path)

    def configure_optimizers(self):
        if self.lr_scheduler:
            # Return both optimizer and scheduler
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": self.lr_scheduler,
            }
        else:
            # Return only the optimizer
            return self.optimizer

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        preds = self.model(inputs)
        if hasattr(self, 'standardizer'):
            preds = self.standardizer(preds, rev=True)
        return preds
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self._step(batch, stage="train")

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self._step(batch, stage="val")
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self._step(batch, stage="test")

    ########### Private Methods ##########################################################
    def _step(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor], 
        stage: Literal['train', 'val', 'test'],
    ) -> torch.Tensor:
        """
        Perform a training, validation, or test step.
        """
        inputs, targets = batch
        preds = self.forward(inputs)
        loss = self.loss(preds, targets)
        # TODO: Log correctly
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        if self.metrics and stage in self.metrics:
            self.log_dict(self.metrics[stage], on_step=False, on_epoch=True)

        return loss
    
    def _init_metrics(self, metrics: MetricCollection | Dict[str, MetricCollection]) -> Dict[str, MetricCollection]:
        if isinstance(metrics, MetricCollection):
            return {
                "val": metrics.clone(prefix="val_"),
                "test": metrics.clone(prefix="test_"),
            }
        elif isinstance(metrics, dict):
            if not all(isinstance(v, MetricCollection) for v in metrics.values()):
                raise TypeError("Metrics must be instances of torchmetrics.MetricCollection.")
            if not all(k in ["train", "val", "test"] for k in metrics.keys()):
                raise ValueError("Metric dictionary keys must be any of 'train', 'val', and 'test'.")
            if len(metrics) != len(set(metrics.keys())):
                raise ValueError("Metric dictionary keys must be unique.")
            return metrics
        else:
            raise TypeError("Metrics must be a torchmetrics.MetricCollection or a dictionary of MetricCollections.")

    def _validate_metrics(self, metrics: Dict[str, Metric]):
        """
        Assert that the metrics are a dictionary of Metric objects.
        """
        for name, metric in metrics.items():
            if not isinstance(metric, Metric):
                raise TypeError(f"Metric '{name}' is not an instance of torchmetrics.Metric.")

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