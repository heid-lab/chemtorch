from abc import ABC, abstractmethod
from typing import Callable, Generic, TypeVar, get_args


T = TypeVar("T")


class TransformBase(ABC, Generic[T]):
    """
    Abstract base class for all stateless transforms in DeepReaction.

    Subclasses must implement the forward method, which takes and returns an object of type T.
    The __call__ method checks the input type at runtime and then calls forward.

    Raises:
        TypeError: If the subclass does not implement the :attr:`forward` method.
        ValueError: If the input data type does not match the expected type T.

    Example (correct usage):
        >>> class MyTransform(TransformBase[int]):
        ...     def forward(self, data: int) -> int:
        ...         return data * 2
        >>> t = MyTransform()
        >>> t(3)
        6

    Example (incorrect usage, raises TypeError):
        >>> class BadTransform(TransformBase[int]):
        ...     pass
        >>> t = BadTransform()
        Traceback (most recent call last):
            ...
        TypeError: Can't instantiate abstract class BadTransform with abstract method forward
    """

    @abstractmethod
    def forward(self, data: T) -> T:
        pass

    def __call__(self, data: T) -> T:
        self._validate_type(data)
        return self.forward(data)

    def _validate_type(self, data):
        expected_type = get_args(self.__orig_class__)[0] if hasattr(self, "__orig_class__") else None
        if expected_type and not isinstance(data, expected_type):
            raise ValueError(f"Input data must be of type {expected_type}, got {type(data)}.")