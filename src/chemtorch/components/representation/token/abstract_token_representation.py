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
    
    @abstractmethod
    def tokenize(self, input: str) -> List[str]:
        """Tokenize the input text into a list of tokens."""
        pass
    
    @abstractmethod
    def extend_vocab(self, new_tokens: List[str]) -> None:
        """
        Extend the vocabulary with new tokens.
        
        Args:
            new_tokens: List of new tokens to add to vocabulary
        """
        pass
    
    @abstractmethod
    def save_vocab(self, vocab_path: str) -> None:
        """
        Save the current vocabulary to a file.
        
        Args:
            vocab_path: Path where to save the vocabulary file
        """
        pass
