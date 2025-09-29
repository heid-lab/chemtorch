from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Union


T = TypeVar("T", covariant=True)


class AbstractRepresentation(ABC, Generic[T]):
    """
    Abstract base class for all chemistry representation creators.

    All representations in ChemTorch must subclass this class and implement the
    `construct` method. This class can also be used for type hinting where
    `AbstractRepresentation[T]` means that the `__call__` and `construct` methods
    will return an object of type `T`.

    The representation should be stateless - it should not hold any mutable state
    and the same input should always produce the same output.

    Raises:
        TypeError: If the subclass does not implement the `construct` method.

    Example (correct usage):
        >>> class MyRepresentation(AbstractRepresentation[torch.Tensor]):
        ...     def construct(self, smiles: str) -> torch.Tensor:
        ...         # Convert SMILES to tensor representation
        ...         return torch.tensor([1, 2, 3])
        >>> r = MyRepresentation()
        >>> r("CCO")  # ethanol
        tensor([1, 2, 3])

    Example (incorrect usage, raises TypeError):
        >>> class BadRepresentation(AbstractRepresentation[torch.Tensor]):
        ...     pass
        >>> r = BadRepresentation()
        Traceback (most recent call last):
            ...
        TypeError: Can't instantiate abstract class BadRepresentation with abstract method construct
    """
    
    @abstractmethod
    def construct(self, smiles: str, **kwargs) -> T:
        """
        Construct a representation from a SMILES string.
        
        Args:
            smiles (str): A SMILES string representing a molecule or reaction.
                For reactions, typically in the format "reactants>reagents>products"
                or "reactants>>products" (e.g., "CCO>>CC=O").
                For molecules, a standard SMILES string (e.g., "CCO").
            **kwargs: Additional keyword arguments that may be required by specific
                representation implementations (e.g., reaction_dir for Reaction3DGraph).
        
        Returns:
            T: The constructed representation of the specified type.
                Common types include torch.Tensor for token representations,
                torch_geometric.data.Data for graph representations, etc.
        
        Raises:
            ValueError: If the SMILES string is invalid or cannot be processed.
            RuntimeError: If representation construction fails for any other reason.
        """
        pass

    def __call__(self, smiles: str, **kwargs) -> T:
        """
        Callable interface for construct method.
        
        Args:
            smiles (str): A SMILES string representing a molecule or reaction.
            **kwargs: Additional keyword arguments to pass to the construct method.
        
        Returns:
            T: The constructed representation.
        """
        return self.construct(smiles, **kwargs)