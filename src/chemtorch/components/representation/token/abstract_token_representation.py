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
    def word2id(self) -> Dict[str, int]:
        """Dictionary mapping tokens to IDs."""
        pass
    
    @property
    @abstractmethod 
    def id2word(self) -> Dict[int, str]:
        """Dictionary mapping IDs to tokens."""
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        pass
    