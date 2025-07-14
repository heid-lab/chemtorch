import torch
from typing_extensions import override

from chemtorch.routine.lightning_routine.supervised_routine import (
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

    def __init__(self, standardizer: Standardizer = None, *args, **kwargs):
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
        preds = self.model(inputs)
        preds = preds.squeeze(-1) if preds.ndim > 1 else preds
        if self.standardizer:
            preds = self.standardizer.destandardize(preds)
        return preds
