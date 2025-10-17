"""
Shared presets and constants for integration tests.

This module provides the single source of truth for test configuration values.
All paths, timeouts, and default overrides are defined here to avoid duplication.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
CONFIG_ROOT = PROJECT_ROOT / "conf"
TEST_DIR = Path(__file__).parent.parent

# Test timeouts (in seconds)
# These serve as defaults for baseline configurations and can be overridden per-model
PRECOMPUTE_TIMEOUT = 30  # Default timeout for precompute/data loading phase
DEFAULT_TIMEOUT = 60     # Default timeout per training epoch

# Common test overrides (applied to all tests unless overridden)
BASE_OVERRIDES = [
    "++log=false",
    "++data_module.subsample=0.01",
    "++trainer.enable_checkpointing=false",
    "++trainer.max_epochs=1",  # Single epoch for quick testing
]
