from typing import Any, Callable, Sequence


class CallableCompose:
    """
    CallableCompose is a callable class that takes a sequence of callabels
    and applies them in order to an input object.
    It is used to compose multiple callables into a single callable object.
    """
    def __init__(self, callables: Sequence[Callable]) -> None:
        """
        Initializes the CallableCompose.

        Args:
            callables (Sequence[Callable]): A sequence of callables to be composed.
        """
        self.callables = callables

    def __call__(self, *args, **kwargs) -> Any:
        """

        Args:
            *args: Positional arguments to be passed to the first callable.
            **kwargs: Keyword arguments to be passed to the first callable.

        Returns:
            Any: The final output after applying all callables.
        """
        if not self.callables:
            raise ValueError("No callables provided to CallableCompose.")
        for idx, c in enumerate(self.callables):
            if not callable(c):
                raise TypeError(f"Element at position {idx} is not callable: {c}")
            try:
                if idx == 0:
                    res = c(*args, **kwargs)
                else:
                    res = c(res)
            except Exception as e:
                raise RuntimeError(f"Error in callable at position {idx}: {e}") from e
        return res