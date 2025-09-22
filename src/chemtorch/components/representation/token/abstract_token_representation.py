from abc import ABC, abstractmethod
from typing import Dict, List
import torch

from chemtorch.components.representation.abstract_representation import AbstractRepresentation


class AbstractTokenRepresentation(AbstractRepresentation[torch.Tensor], ABC):
    """
    Abstract base class for token-based representations.
    
    All token representations must implement vocabulary management methods.
    """
    # Abstract properties - subclasses must implement these

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        pass
    