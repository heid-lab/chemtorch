"""Utilities for loading and accessing baseline reference data."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import yaml

from .presets import DEFAULT_TIMEOUT, PRECOMPUTE_TIMEOUT


@dataclass
class ModelBaseline:
    """Baseline configuration for a single model."""
    name: str
    val_loss: Dict[str, float]  # e.g., {"epoch_1": 1.34, "epoch_3": 1.39}
    timeout: Dict[str, int]  # e.g., {"precompute": 10, "per_epoch": 60}
    reference_predictions: Dict[str, str]  # e.g., {"epoch_1": "file.csv"}
    
    def get_val_loss(self, num_epochs: int) -> Optional[float]:
        """Get expected validation loss for a given epoch count."""
        return self.val_loss.get(f"epoch_{num_epochs}")
    
    def get_reference_prediction_file(self, num_epochs: int) -> Optional[str]:
        """Get reference prediction filename for a given epoch count."""
        return self.reference_predictions.get(f"epoch_{num_epochs}")
    
    def calculate_timeout(self, num_epochs: int) -> int:
        """Calculate total timeout for a given number of epochs."""
        return self.timeout["precompute"] + self.timeout["per_epoch"] * num_epochs


class BaselineConfig:
    """Container for all baseline configurations."""
    
    def __init__(self, config_path: Path):
        """
        Load baseline configuration from YAML file.
        
        Args:
            config_path: Path to the baselines.yaml file
        """
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Load defaults from YAML, falling back to constants
        self.defaults = data.get("defaults", {})
        self.tolerance = self.defaults.get("tolerance", 1e-5)
        
        # Get default timeout values (from YAML or constants)
        default_precompute = self.defaults.get("precompute_timeout", PRECOMPUTE_TIMEOUT)
        default_per_epoch = self.defaults.get("timeout_per_epoch", DEFAULT_TIMEOUT)
        
        # Parse model configurations
        self._models: Dict[str, ModelBaseline] = {}
        for model_name, model_data in data.get("models", {}).items():
            # Use model-specific timeouts, or fall back to defaults
            timeout_data = model_data.get("timeout", {})
            timeout = {
                "precompute": timeout_data.get("precompute", default_precompute),
                "per_epoch": timeout_data.get("per_epoch", default_per_epoch),
            }
            
            self._models[model_name] = ModelBaseline(
                name=model_name,
                val_loss=model_data.get("val_loss", {}),
                timeout=timeout,
                reference_predictions=model_data.get("reference_predictions", {}),
            )
    
    def has_baseline(self, model_name: str) -> bool:
        """Check if a baseline exists for the given model."""
        return model_name in self._models
    
    def get_model(self, model_name: str) -> Optional[ModelBaseline]:
        """Get baseline configuration for a model."""
        return self._models.get(model_name)
    
    def list_models(self) -> list[str]:
        """List all models with baseline configurations."""
        return list(self._models.keys())


# Singleton instance for easy access
_baseline_config: Optional[BaselineConfig] = None


def load_baselines(config_path: Path) -> BaselineConfig:
    """Load baseline configuration (singleton pattern)."""
    global _baseline_config
    if _baseline_config is None:
        _baseline_config = BaselineConfig(config_path)
    return _baseline_config


def get_baselines() -> Optional[BaselineConfig]:
    """Get the loaded baseline configuration."""
    return _baseline_config
