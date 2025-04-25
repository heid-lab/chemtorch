from abc import ABC, abstractmethod
from typing import Any, List, Optional


class FeaturizerBase(ABC):
    """Base class for all featurizers."""
    
    @abstractmethod
    def __call__(self, item: Optional[Any]) -> List[float]:
        """Generate features for the given item.
        
        Args:
            item: Item to featurize (Atom, Bond, etc.) or None
            
        Returns:
            List of numerical features
        """
        pass
