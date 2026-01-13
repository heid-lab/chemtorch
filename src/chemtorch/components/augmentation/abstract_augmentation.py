from abc import ABC, abstractmethod
from typing import Generic, List, Optional, TypeVar
import torch

T = TypeVar("T")

class AbstractAugmentation(ABC, Generic[T]):
    """
    Abstract base class for data augmentations in the chemtorch framework.

    This class serves as a base for creating data augmentations that operate on data points.
    A data point (or sample) is an object of type T returned by the representation and an
    optional label tensor.

    Subclasses must implement the :attr:`__call__` method to define the augmentation logic.

    Raises:
        TypeError: If the subclass does not implement the :attr:`__call__` method.
        RuntimeError: If the input object has a label, but not all augmented objects have labels, or vice versa.
    """

    def __call__(self, obj: T, label: Optional[torch.Tensor] = None) -> List[tuple[T, Optional[torch.Tensor]]]:
        """
        Abstract method to be implemented by subclasses.
        This method should define the augmentation logic.

        Args:
            obj (T): The data object to be augmented.
            label (Optional[torch.Tensor]): The optional label tensor.

        Returns:
            List[Tuple[T, Optional[torch.Tensor]]]: A list of augmented objects with optional labels.
        """
        has_label = label is not None
        augmented = self._augment(obj=obj, label=label)
        if has_label and any(lab is None for _, lab in augmented):
            raise RuntimeError("If the input object has a label, all augmented objects must also have labels.")
        if not has_label and any(lab is not None for _, lab in augmented):
            raise RuntimeError("If the input object does not have a label, all augmented objects must also not have labels.")
        return augmented

    @abstractmethod
    def _augment(self, obj: T, label: Optional[torch.Tensor] = None) -> List[tuple[T, Optional[torch.Tensor]]]:
        """
        Internal method to be implemented by subclasses.
        This method should define the augmentation logic.

        Args:
            obj (T): The data object to be augmented.
            label (Optional[torch.Tensor]): The optional label tensor.

        Returns:
            List[Tuple[T, Optional[torch.Tensor]]]: A list of augmented objects with optional labels.

        Note: Do not modify the input object in-place. If modifications are needed, work on 
        a clone or deepcopy of the input object.
        """
        pass
