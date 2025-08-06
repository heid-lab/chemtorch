import torch
try:
    # Python ≥ 3.12
    from typing import override  # type: ignore
except ImportError:
    # Python < 3.12
    from typing_extensions import override  # type: ignore
from typing import Any, Tuple, Literal

from torchmetrics import Metric, MetricCollection
from chemtorch.routine.supervised_routine import (
    SupervisedRoutine,
)
from chemtorch.utils.standardizer import Standardizer


class RegressionRoutine(SupervisedRoutine):
    """
    Extends SupervisedRoutine for regression tasks by allowing the use of an optional standardizer.

    This class is intended for regression models where outputs may need to be destandardized
    (e.g., to return predictions in the original scale). If a `Standardizer` is provided,
    predictions are automatically destandardized in the forward pass.

    Args:
        standardizer (Standardizer, optional): An instance of Standardizer for output destandardization.
        *args: Additional positional arguments for SupervisedRoutine.
        **kwargs: Additional keyword arguments for SupervisedRoutine.

    See Also:
        :class:`chemtorch.routine.supervised_routine.SupervisedRoutine`

    Examples:
        >>> # With standardizer (for regression)
        >>> routine = RegressionRoutine(
        ...     model=my_model,
        ...     standardizer=my_standardizer,
        ...     loss=my_loss_fn,
        ...     optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
        ...     metrics=my_metrics,
        ... )
        >>> preds = routine(torch.randn(8, 16))  # Returns destandardized predictions

        >>> # Without standardizer (raw model output)
        >>> routine = RegressionRoutine(
        ...     model=my_model,
        ...     loss=my_loss_fn,
        ...     optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
        ...     metrics=my_metrics,
        ... )
        >>> preds = routine(torch.randn(8, 16))  # Returns raw model predictions
    """

    def __init__(self, standardizer: Standardizer | None = None, *args, **kwargs):
        """
        Initialize a regression routine with an optional standardizer.

        Args:
            standardizer (Standardizer, optional): An instance of Standardizer for data normalization.
            *args: Additional positional arguments for SupervisedRoutine.
            **kwargs: Additional keyword arguments for SupervisedRoutine.
        """
        super().__init__(*args, **kwargs)
        self.standardizer = standardizer

    @override
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        preds = super().forward(inputs)
        preds = preds.squeeze(-1) if preds.ndim > 1 else preds
        return preds

    @override
    def predict_step(self, *args: Any, **kwargs: Any) -> Any:
        preds = super().predict_step(*args, **kwargs)
        if self.standardizer:
            preds = self.standardizer.destandardize(preds)
        return preds
    
    @override
    def _loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.standardizer:
            targets = self.standardizer.standardize(targets)
        return super()._loss(preds, targets)

    @override
    def _update_metrics(self, metrics: Metric | MetricCollection, preds: torch.Tensor, targets: torch.Tensor):
        if self.standardizer:
            preds = self.standardizer.destandardize(preds)
        return super()._update_metrics(metrics, preds, targets)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self.standardizer:
            checkpoint['standardizer_mean'] = self.standardizer.mean
            checkpoint['standardizer_std'] = self.standardizer.std

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if 'standardizer_mean' in checkpoint and 'standardizer_std' in checkpoint:
            from chemtorch.utils.standardizer import Standardizer
            self.standardizer = Standardizer(
                mean=checkpoint['standardizer_mean'],
                std=checkpoint['standardizer_std']
            )
