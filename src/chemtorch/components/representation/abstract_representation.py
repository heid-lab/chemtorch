from abc import ABC, abstractmethod
from typing import TypeVar, Generic


T = TypeVar("T")


class AbstractRepresentation(ABC, Generic[T]):
    """
    Abstract base class for all stateless representation creators.

    Subclasses should implement `forward` with the arguments they require.
    The only requirement is that the return type is compatible with the dataset and transforms.

    Raises:
        TypeError: If the subclass does not implement the :attr:`forward` method.

    Example (correct usage):
        >>> class MyRepresentation(RepresentationBase[int]):
        ...     def construct(self, a: int, b: int) -> int:
        ...         return a + b
        >>> r = MyRepresentation()
        >>> r(1, 2)
        3

    Example (incorrect usage, raises TypeError):
        >>> class BadRepresentation(RepresentationBase[int]):
        ...     pass
        >>> r = BadRepresentation()
        Traceback (most recent call last):
            ...
        TypeError: Can't instantiate abstract class BadRepresentation with abstract method forward
    """
    

    def __call__(self, *args, **kwargs) -> T:
        return self.construct(*args, **kwargs)


    @abstractmethod
    def construct(self, *args, **kwargs) -> T:
        """
        Construct the representation object from input arguments.
        This method should be implemented by subclasses.

        Args:
            *args: Positional arguments for constructing the representation.
            **kwargs: Keyword arguments for constructing the representation.

        Returns:
            T: The constructed representation object.
        """
        pass