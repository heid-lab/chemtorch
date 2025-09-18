from abc import ABC, abstractmethod
from typing import List


class AbstractTokenizer(ABC):
    """Abstract base class for tokenizers."""

    @abstractmethod
    def tokenize(self, smiles: str) -> List[str]:
        """Tokenize the input SMILES string into a list of tokens."""
        pass