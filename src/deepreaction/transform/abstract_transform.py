from abc import ABC, abstractmethod
from typing import Generic, TypeVar, get_args


T = TypeVar("T")


class AbstractTransform(ABC, Generic[T]):
    """
    Abstract base class for transforms in the DeepReaction framework.
    This class serves as a base for creating transforms that operate on single data 
    points.

    Raises:
        TypeError: If the subclass does not implement the :attr:`__call__` method.

    Example (correct usage):
        >>> class MyTransform(TransformBase[int]):
        ...     def __call__(self, data: int) -> int:
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
        TypeError: Can't instantiate abstract class BadTransform with abstract method __call__
    """

    @abstractmethod
    def __call__(self, data: T) -> T:
        pass