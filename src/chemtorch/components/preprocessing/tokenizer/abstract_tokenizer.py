from abc import ABC, abstractmethod
from typing import List


class AbstractTokenizer(ABC):
    """Abstract base class for tokenizers."""

    @abstractmethod
    def tokenize(self, smiles: str) -> List[str]:
        """Tokenize the input SMILES string into a list of tokens. If the input does not yield any tokens, return an empty list."""
        pass

    @property
    @abstractmethod
    def vocab_path(self) -> str:
        """Path to the vocabulary file."""
        pass

    @property
    @abstractmethod
    def unk_token(self) -> str:
        """Token to use for unknown tokens."""
        pass

    @property
    @abstractmethod
    def pad_token(self) -> str:
        """Token to use for padding."""
        pass
