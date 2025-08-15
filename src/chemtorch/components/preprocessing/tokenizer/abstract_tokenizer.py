from abc import ABC, abstractmethod
from typing import List


class AbstractTokenizer(ABC):
    """Abstract base class for tokenizers."""
    
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize the input text into a list of tokens."""
        pass