from typing import Dict, Tuple, Callable, Iterator, Literal, Union, Mapping, Any

import os
import warnings
import torch
import lightning as L

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import Metric, MetricCollection


class SupervisedRoutine(L.LightningModule):
    """
    A flexible LightningModule wrapper for supervised tasks, supporting both training and inference.

    This class can be used for:
      - Full training/validation/testing with loss, optimizer, scheduler, and metrics.
      - Inference-only (prediction), requiring only the model.

    Example usage:

        >>> # Training usage
        >>> routine = SupervisedRoutine(
        ...     model=my_model,
        ...     loss=my_loss_fn,
        ...     optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
        ...     lr_scheduler=lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=10),
        ...     metrics=my_metrics,
        ... )
        >>> trainer = pl.Trainer(...)
        >>> trainer.fit(routine, datamodule=my_datamodule)

        >>> # Inference-only usage
        >>> routine = SupervisedRoutine(model=my_model)
        >>> preds = routine(torch.randn(8, 16))  # Forward pass for prediction
    """
    def __init__(
            self, 
            model: nn.Module, 
            loss: Callable = None, 
            optimizer: Union[Callable[[Iterator[nn.Parameter]], Optimizer], None] = None,
            lr_scheduler: Union[Callable[[Optimizer], LRScheduler], Dict[str, Any], None] = None,
            ckpt_path: str = None,
            resume_training: bool = False,
            metrics: Union[Metric, MetricCollection, Dict[str, Union[Metric, MetricCollection]]] = None,
        ):
        """
        Initialize the SupervisedRoutine.
        
        Args:
            model (nn.Module): The model to be trained or used for inference.
            loss (Callable, optional): The loss function to be used. Required for training/validation/testing.
            optimizer (Callable, optional): A factory function that takes in the model's parameters 
                and returns an optimizer instance. Required for training/validation/testing.
                
                Example:
                    optimizer=lambda params: torch.optim.Adam(params, lr=1e-3)
                    
            lr_scheduler (Callable or Dict, optional): Either a factory function that takes in the optimizer
                and returns a learning rate scheduler instance, or a Lightning config dictionary containing
                a "scheduler" key with the partially instantiated scheduler factory and optional Lightning-specific keys.
                Only needed for training.
                
                Examples:
                    # Factory function approach
                    lr_scheduler=lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=10)
                    
                    # Lightning config dictionary approach
                    lr_scheduler={
                        "scheduler": partial_scheduler_factory,  # e.g., functools.partial(torch.optim.lr_scheduler.StepLR, step_size=10)
                        # Lightning-specific keys (optional):
                        "interval": "epoch",
                        "frequency": 1,
                        "monitor": "val_loss",
                        # etc.
                    }
                    
            ckpt_path (str, optional): Path to a pre-trained model checkpoint.
            resume_training (bool, optional): Whether to resume training from a checkpoint.
            metrics (Metric, MetricCollection or Dict[str, Metric/MetricCollection], optional): Metrics to use for evaluation.
                - If a single `Metric` is provided, it will be cloned for 'train', 'val' and 'test' stages.
                - If a single `MetricCollection` is provided, it will be cloned for 'train', 'val' and 'test' stages.
                - If a dictionary is provided, it must map keys 'train', 'val', and/or 'test' to 
                `Metric` or `MetricCollection` instances. This allows you to specify different metrics for each stage.
                In all cases, the metrics will be registered as attributes of the LightningModule for proper logging.

                Example usage:
                    >>> from torchmetrics import MetricCollection, MeanAbsoluteError, MeanSquaredError
                    ...
                    >>> # Single Metric for all stages
                    >>> metric = MeanAbsoluteError()
                    >>> routine = SupervisedRoutine(
                    ...     model=my_model,
                    ...     loss=my_loss_fn,
                    ...     optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
                    ...     metrics=metric,
                    ... )
                    >>> # Single MetricCollection for all stages
                    >>> metrics = MetricCollection({
                    ...     "mae": MeanAbsoluteError(),
                    ...     "rmse": MeanSquaredError(squared=False),
                    ... })
                    >>> routine = SupervisedRoutine(
                    ...     model=my_model,
                    ...     loss=my_loss_fn,
                    ...     optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
                    ...     metrics=metrics,
                    ... )
                    >>> # Distinct metrics for each stage (mix of single metrics and collections)
                    >>> metrics_dict = {
                    ...     "train": MeanAbsoluteError(),  # Single metric
                    ...     "val": MetricCollection({"rmse": MeanSquaredError(squared=False)}),  # Collection
                    ...     "test": MeanSquaredError(),  # Single metric
                    ... }
                    >>> routine = SupervisedRoutine(
                    ...     model=my_model,
                    ...     loss=my_loss_fn,
                    ...     optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
                    ...     metrics=metrics_dict,
                    ... )

        Raises:
            TypeError: If `metrics` is not a Metric, MetricCollection, or a dictionary of Metrics/MetricCollections.
            ValueError: If `metrics` is a dictionary, but its keys are not 'train', 'val', or 'test',
                or if the keys are not unique.
        """
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer_factory = optimizer
        self.lr_scheduler_config = lr_scheduler
        self.metrics = self._init_metrics(metrics) if metrics and (not isinstance(metrics, dict) or len(metrics) > 0) else None
        self.ckpt_path = ckpt_path
        self.resume_training = resume_training
        self.metrics = self._init_metrics(metrics) if self._should_init_metrics(metrics) else None
        self.ckpt_path = ckpt_path
        self.resume_training = resume_training

    def _should_init_metrics(self, metrics) -> bool:
        """
        Helper to determine if metrics should be initialized.
        Returns True if metrics is not None and either not a dict or a non-empty dict.
        """
        if metrics is None:
            return False
        if isinstance(metrics, dict):
            return len(metrics) > 0
        return True
    ########## LightningModule Methods ##############################################
    def setup(self, stage: Literal['fit', 'validate', 'test', 'predict'] | None = None):
        if stage in ['fit', 'validate', 'test']:
            if self.loss is None:
                raise ValueError("Loss function must be defined for training.")
            if self.optimizer_factory is None:
                raise ValueError("Optimizer must be defined for training.")
        
        # Handle checkpoint loading
        if self.ckpt_path:
            self._load_pretrained(self.ckpt_path, self.resume_training)
        elif self.resume_training:
            # Resume training requested but no checkpoint provided
            warnings.warn(
                "resume_training=True but no checkpoint path was provided. "
                "Training will start from scratch with random initialization.",
                UserWarning,
                stacklevel=2
            )

    def configure_optimizers(self):
        if self.optimizer_factory is None:
            return None
        
        # Create optimizer from factory function
        optimizer = self.optimizer_factory(self.model.parameters())
        
        # Load optimizer state if resuming training
        if self.resume_training and hasattr(self, '_checkpoint_state') and 'optimizer_state_dict' in self._checkpoint_state:
            optimizer.load_state_dict(self._checkpoint_state['optimizer_state_dict'])
        
        # Handle lr_scheduler configuration
        if self.lr_scheduler_config is None:
            # Return just the optimizer
            return optimizer
        
        if callable(self.lr_scheduler_config):
            # Factory function approach
            lr_scheduler = self.lr_scheduler_config(optimizer)
            lr_scheduler_config = {"scheduler": lr_scheduler}
        elif isinstance(self.lr_scheduler_config, dict):
            # Lightning config dictionary approach
            config = self.lr_scheduler_config.copy()
            if "scheduler" not in config:
                raise ValueError("LR scheduler config dictionary must contain 'scheduler' key with the scheduler factory")
            
            scheduler_factory = config.pop("scheduler")
            lr_scheduler = scheduler_factory(optimizer)
            lr_scheduler_config = {"scheduler": lr_scheduler, **config}
        else:
            raise TypeError(f"LR scheduler must be callable or dict, got {type(self.lr_scheduler_config)}")
        
        # Load scheduler state if resuming training
        if self.resume_training and hasattr(self, '_checkpoint_state') and 'lr_scheduler_state_dict' in self._checkpoint_state:
            lr_scheduler.load_state_dict(self._checkpoint_state['lr_scheduler_state_dict'])
        
        # Return optimizer and scheduler configuration
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self._step(batch, split="train")

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self._step(batch, split="val")
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self._step(batch, split="test")

    ########### Private Methods ##########################################################
    def _step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor], 
        split: Literal['train', 'val', 'test'],
    ) -> torch.Tensor:
        """
        Perform a training, validation, or test step.
        """
        inputs, targets = batch
        batch_size = inputs.size(0)
        preds = self.forward(inputs)
        loss = self._loss(preds, targets)

        self.log(f"{split}_loss", loss, 
            on_step=True, 
            batch_size=batch_size,
            prog_bar = True, 
        )

        if self.metrics and split in self.metrics:
            self._update_metrics(self.metrics[split], preds, targets)
            if isinstance(self.metrics[split], Metric):
                self.log(
                    self.metrics[split]._get_name(), 
                    self.metrics[split], 
                    on_step=False, 
                    on_epoch=True,
                    batch_size=batch_size,
                )
            elif isinstance(self.metrics[split], MetricCollection):
                self.log_dict(
                    self.metrics[split], 
                    on_step=False, 
                    on_epoch=True,
                    batch_size=batch_size
                )

        return loss
    
    def _loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss for the given predictions and targets.
        Override this method to customize loss computation.
        """
        return self.loss(preds, targets)

    def _update_metrics(self, metrics: Metric | MetricCollection, preds: torch.Tensor, targets: torch.Tensor):
        """
        Update the metrics with the current predictions and targets.
        Override this method to customize metric updates.
        """
        metrics.update(preds, targets)
    
    def _init_metrics(self, metrics: Union[Metric, MetricCollection, Dict[str, Union[Metric, MetricCollection]]]) -> Dict[str, Union[Metric, MetricCollection]]:
        metrics_dict = {}
        if isinstance(metrics, Metric):
            # Single metric for all stages - clone and reset state
            for stage in ["train", "val", "test"]:
                cloned_metric = metrics.clone()
                cloned_metric.reset()
                cloned_metric.prefix = f"{stage}_"
                metrics_dict[stage] = cloned_metric
        elif isinstance(metrics, MetricCollection):
            # Single MetricCollection for all stages - clone and reset state
            for stage in ["train", "val", "test"]:
                cloned_collection = metrics.clone(prefix=f"{stage}_")
                cloned_collection.reset()
                metrics_dict[stage] = cloned_collection
        elif isinstance(metrics, dict):
            # Dictionary that assigns each stage its own Metric/MetricCollection
            if not all(isinstance(v, (Metric, MetricCollection)) for v in metrics.values()):
                raise TypeError("Metrics must be instances of torchmetrics.Metric or torchmetrics.MetricCollection.")
            if not all(k in ["train", "val", "test"] for k in metrics.keys()):
                raise ValueError("Metric dictionary keys must be any of 'train', 'val', and 'test'.")
            if len(metrics) != len(set(metrics.keys())):
                raise ValueError("Metric dictionary keys must be unique.")
            
            # Clone metrics and ensure proper prefixes
            for stage, metric in metrics.items():
                if isinstance(metric, Metric):
                    cloned_metric = metric.clone()
                    cloned_metric.reset()
                    cloned_metric.prefix = f"{stage}_"
                    metrics_dict[stage] = cloned_metric
                else:
                    cloned_collection = metric.clone(prefix=f"{stage}_")
                    cloned_collection.reset()
                    metrics_dict[stage] = cloned_collection
        else:
            raise TypeError(f"Metrics must be a torchmetrics.Metric, torchmetrics.MetricCollection, or a dictionary of Metrics/MetricCollections, got {type(metrics)}")


        # Register each metric/MetricCollection as an attribute (required by Lightning for logging)
        for stage, metric in metrics_dict.items():
            setattr(self, f"{stage}_metrics", metric)

        return metrics_dict

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
        # NOTE:PyTorch deserializes checkpoint on CPU and then moves them to the device-type
        # it saved on (e.g. 'cuda') by default. This can lead to surged in memory usage
        # if the model is large. To avoid this, we load the checkpoint on CPU and move 
        # the model to the correct device later.
        # We also use `weights_only=True` to load only the model weights, out of security
        # reasons.

        # Always validate that model state dict is present
        required_keys = ['model_state_dict']
        self._validate_checkpoint_dict(checkpoint, required_keys)

        # Always load the model state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Store checkpoint state for potential optimizer/scheduler loading if resuming training
        if resume_training:
            # When resuming training, we expect optimizer state dict to be present
            if 'optimizer_state_dict' not in checkpoint:
                warnings.warn(
                    "resume_training=True but checkpoint does not contain 'optimizer_state_dict'. "
                    "Optimizer will start with fresh initialization.",
                    UserWarning,
                    stacklevel=2
                )
            if self.lr_scheduler_config and 'lr_scheduler_state_dict' not in checkpoint:
                warnings.warn(
                    "resume_training=True but checkpoint does not contain 'lr_scheduler_state_dict'. "
                    "LR scheduler will start with fresh initialization.",
                    UserWarning,
                    stacklevel=2
                )
            # Store checkpoint for later use in configure_optimizers
            self._checkpoint_state = checkpoint

        self.log("pretrained_model_loaded", True, prog_bar=True)

    def _validate_checkpoint_dict(self, checkpoint: dict, keys: list):
        """
        Validate that the checkpoint dictionary contains the required keys.
        """
        missing_keys = [key for key in keys if key not in checkpoint]
        if missing_keys:
            raise ValueError(f"Checkpoint is missing required keys: {missing_keys}")