from typing import Any, Callable, Generic, Sequence, TypeVar

R = TypeVar("R")
T = TypeVar("T")


class CallableCompose(Generic[R, T]):
    """
    CallableCompose composes a sequence of callables into a single callable object.

    This class is generic over input type R and output type T, and allows you to chain
    multiple callables (functions, transforms, etc.) so that the output of each is passed
    as the input to the next. The composed object itself is callable and applies all
    callables in order.

    Typical use cases include composing data transforms, preprocessing steps, or any
    sequence of operations that should be applied in a pipeline fashion.

    Type parameters:
        R: The input type to the first callable.
        T: The output type of the final callable.

    Example:
        >>> def double(x: int) -> int:
        ...     return x * 2
        >>> def stringify(x: int) -> str:
        ...     return str(x)
        >>> composed = CallableCompose([double, stringify])
        >>> composed(3)
        '6'

    You can also use the static method `compose` for a more concise syntax:

        >>> composed = CallableCompose.compose(double, stringify)
        >>> composed(4)
        '8'
    """
    def __init__(self, callables: Sequence[Callable]) -> None:
        """
        Initializes the CallableCompose.

        Args:
            callables (Sequence[Callable]):
                A sequence of callables to be composed. Each callable should accept the output
                type of the previous callable (the first should accept type R, the last should return type T).
        """
        self.callables = callables

    def __call__(self, obj: R) -> T:
        """
        Apply the composed callables in order to the input object.

        Args:
            obj (R): The input object to be passed to the first callable.

        Returns:
            T: The final output after applying all callables in sequence.
        """
        if not self.callables:
            raise ValueError("No callables provided to CallableCompose.")
        res = obj
        for idx, c in enumerate(self.callables):
            if not callable(c):
                raise TypeError(f"Element at position {idx} is not callable: {c}")
            try:
                res = c(res)
            except Exception as e:
                raise RuntimeError(f"Error in callable at position {idx}: {e}") from e
        return res  # type: ignore

    @staticmethod
    def compose(*callables: Callable[[T], T]) -> "CallableCompose[T, T]":
        """
        Convenience method to compose multiple callables of the same input/output type.

        Args:
            *callables: Callables to compose, each of type Callable[[T], T].

        Returns:
            CallableCompose[T, T]: A composed callable applying all in order.
        """
        return CallableCompose(list(callables))