import pandas as pd
import pytest

from chemtorch.components.representation.abstract_representation import AbstractRepresentation
from chemtorch.components.transform.abstract_transform import AbstractTransform


class NoOpMockRepresentation(AbstractRepresentation[None]):
    """Mock representation that returns None for testing purposes."""
    def construct(self, smiles: str) -> None:
        return None


class MockTransform(AbstractTransform[None]):
    """Mock transform that adds a marker to test transform application."""
    def __init__(self, marker: str = "transformed"):
        self.marker = marker
        self.call_count = 0
    
    def __call__(self, obj: None) -> str:
        self.call_count += 1
        return f"{self.marker}_{self.call_count}"


class CallableTransform:
    """Mock callable transform (not inheriting from AbstractTransform)."""
    def __init__(self, prefix: str = "callable"):
        self.prefix = prefix
        self.call_count = 0
    
    def __call__(self, obj: None) -> str:
        self.call_count += 1
        return f"{self.prefix}_{self.call_count}"


@pytest.fixture
def train_df():
    """Sample training dataframe for testing."""
    return pd.DataFrame({"smiles": ["A", "B", "C", "D", "E"], "label": [1, 2, 3, 4, 5]})


@pytest.fixture
def val_df():
    """Sample validation dataframe for testing."""
    return pd.DataFrame({"smiles": ["F", "G"], "label": [6, 7]})


@pytest.fixture
def test_df():
    """Sample test dataframe for testing."""
    return pd.DataFrame({"smiles": ["H", "I"], "label": [8, 9]})


@pytest.fixture
def mock_representation():
    """Fixture providing a mock representation instance."""
    return NoOpMockRepresentation()


@pytest.fixture
def mock_transform():
    """Fixture providing a mock transform instance."""
    return MockTransform()


@pytest.fixture
def callable_transform():
    """Fixture providing a callable transform instance."""
    return CallableTransform()