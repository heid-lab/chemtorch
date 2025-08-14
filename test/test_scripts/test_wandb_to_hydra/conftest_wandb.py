"""
Configuration and shared fixtures for wandb_to_hydra tests.
"""
import pytest
import sys
from pathlib import Path

def pytest_configure():
    """Configure pytest to add the scripts directory to the Python path."""
    # Add the scripts directory to sys.path so we can import the module under test
    project_root = Path(__file__).parent.parent
    scripts_dir = project_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

@pytest.fixture
def project_root():
    """Fixture that provides the project root directory."""
    return Path(__file__).parent.parent

@pytest.fixture
def scripts_dir(project_root):
    """Fixture that provides the scripts directory."""
    return project_root / "scripts"

@pytest.fixture
def wandb_to_hydra_script(scripts_dir):
    """Fixture that provides the path to the wandb_to_hydra.py script."""
    return scripts_dir / "wandb_to_hydra.py"
