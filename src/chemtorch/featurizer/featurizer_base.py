from typing import Any, Callable, Generic, List, Optional, Tuple, TypeVar

from chemtorch.featurizer.abstract_featurizer import AbstractFeaturizer
from chemtorch.featurizer.utils import featurize
from chemtorch.utils import enforce_base_init

T = TypeVar("T")


class FeaturizerBase(AbstractFeaturizer, Generic[T]):
    """
    Generic base class for featurizer implementations.

    This class implements the AbstractFeaturizer interface and encapsulates common featurization
    logic. It is designed to be subclassed by concrete featurizer implementations.
    Subclasses must call `super().__init__(features)` in their `__init__` method and pass
    a list of implementation-specific `features`.
    """

    def __init__(
        self, features: List[Callable[[T], Any] | Tuple[Callable[[T], Any], List]]
    ):
        """
        Initialize the featurizer with a list of features.

        Args:
            features (List[Callable | Tuple[Callable, List]]): A list of features to be extracted. Please
                refer to `chemtorch.featurizer.utils.featurize` for details on the expected format.
        """
        self.features = features

    def __init_subclass__(cls) -> None:
        enforce_base_init(FeaturizerBase)(cls)
        return super().__init_subclass__()

    def __call__(self, item: Optional[T]) -> List[float]:
        """
        Featurize the given item using the features defined in this featurizer.

        Args:
            item (Optional[T]): The item to be featurized. If `None`, a list of zeros will be returned.

        Returns:
            List[float]: The featurized representation of the item as a list of floats.
        """
        return featurize(item, self.features)
