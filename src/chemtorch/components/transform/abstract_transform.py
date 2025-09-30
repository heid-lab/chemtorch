from abc import ABC, abstractmethod
from typing import Generic, TypeVar


T = TypeVar("T")


class AbstractTransform(ABC, Generic[T]):
    """
    Abstract base class for transforms in the chemtorch framework.
    This class serves as a base for creating transforms that operate on single data objects.

    Raises:
        TypeError: If the subclass does not implement the :attr:`__call__` method.

    Example (correct usage):
        >>> class MyTransform(TransformBase[int]):
        ...     def __call__(self, obj: int) -> int:
        ...         return obj * 2
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
    def __call__(self, obj: T) -> T:
        """
        Abstract method to be implemented by subclasses.
        This method should define the transformation logic.

        Args:
            obj (T): The object to be transformed.

        Returns:
            T: The transformed object.
        """
        pass
