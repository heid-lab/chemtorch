from typing import Callable, Optional, TypeVar, Generic
import pandas as pd

from deepreaction.representation.representation_base import RepresentationBase
from deepreaction.transform.abstract_transform import AbstractTransform
from deepreaction.utils.decorators.enforce_base_init import enforce_base_init


T = TypeVar("T")


class DatasetBase(Generic[T]):
    """
    Base class for DeepReaction datasets.

    This class defines the standard interface for datasets in the DeepReaction framework.
    All datasets should subclass :class:`DatasetBase[T]` and implement the `_get_sample_by_idx` method.    

    Raises:
        RuntimeError: If the subclass does not call `super().__init__()` in its `__init__()` method.

    Example:
        >>> import pandas as pd
        >>> class MyRepresentation(RepresentationBase[int]):
        ...     def __call__(self, a: int, b: int) -> int:
        ...         return a + b
        ...
        >>> class MyTransform(TransformBase[int]):
        ...     def __call__(self, data: int) -> int:
        ...         return data * 2
        ...
        >>> df = pd.DataFrame([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        >>> dataset = DatasetBase(df, MyRepresentation, MyTransform)
        >>> result = dataset._process_sample_by_idx(0)
        >>> print(result)
        6
    """
    

    def __init__(
        self,
        dataframe: pd.DataFrame,
        representation: RepresentationBase[T] | Callable[..., T],
        transform: Optional[AbstractTransform[T] | Callable[[T], T]] = None
    ):
        """
        Initialize the DatasetBase.

        Args:
            dataframe (pd.DataFrame): The input data as a pandas DataFrame. Each row represents a single sample.
            representation (RepresentationBase[T] | Callable[..., T]): A stateless class or callable that 
                constructs the data object consumed by the model. Must take in the fields of a single sample 
                from the :attr:`dataframe` (row) as keyword arguments and return an object of type T.
            transform (Optional[TransformBase[T] | Callable[[T], T]]): An optional transformation function or a composition thereof 
                (:class:`Compose`) that takes in an object of type T and returns a (possibly modified) object of 
                the same type.

        Raises:
            ValueError: If the `dataframe` is not a pandas DataFrame.
            ValueError: If the `representation` is not a RepresentationBase or a callable.
            ValueError: If the `transform` is not a TransformBase, a callable, or None.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("dataframe must be a pandas DataFrame.")
        if not isinstance(representation, (RepresentationBase, Callable)):
            raise ValueError(
                "representation must be a RepresentationBase or a callable."
            )
        if not isinstance(transform, (AbstractTransform, Callable, type(None))):
            raise ValueError(
                "transform must be a TransformBase, a callable, or None."
            )

        self.dataframe = dataframe
        self.representation = representation
        self.transform = transform

        self._initialized_by_base = True  # mark successful call


    def __init_subclass__(cls):
        enforce_base_init(DatasetBase)(cls)
        return super().__init_subclass__()
    

    def _process_sample(self, sample: pd.Series | dict) -> T:
        """
        Process a sample by its index.

        This method uses the representation callable to create a representation data
        object from the sample data. If a transform is provided, it applies the transform 
        to the representation object.

        Args:
            sample (pd.Series | dict): The sample data to process. It can be a pandas Series or a dictionary.
        
        Returns:
            T: The processed representation object.

        Raises:
            ValueError: If the sample is not a pandas Series or a dictionary.
        """
        if isinstance(sample, pd.Series):
            sample = sample.to_dict()
        elif not isinstance(sample, dict):
            raise ValueError(f"Sample must be a pandas Series or a dictionary. Received: {type(sample)}")
        data_obj = self.representation(**sample)
        if self.transform:
            data_obj = self.transform(data_obj)
        return data_obj
    
    # TODO: Add abstract method _get_sample_by_idx to guide implementation?


