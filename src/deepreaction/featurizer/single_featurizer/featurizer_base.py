from abc import ABC, abstractmethod
from typing import Any, List, Optional


class FeaturizerBase(ABC):
    """Base class for all featurizers."""
    
    @abstractmethod
    def __call__(self, atom: Optional[Any]) -> List[float]:
        """Generate features for the given item.
        
        Args:

            
        Returns:
            List of numerical features
        """
        pass
