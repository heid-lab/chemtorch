"""Base class for config testers with common functionality."""

import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from time import time
from typing import List, Optional, Tuple

from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
import pytest

from .presets import BASE_OVERRIDES, DEFAULT_TIMEOUT


class ConfigTester(ABC):
    """
    Abstract base class for testing Hydra configurations.
    
    Subclasses must implement:
    - _init_config: How to initialize a config
    - _build_base_cmd: How to build the command to run the config
    """
    
    def __init__(
        self,
        config_dir_from_config_root: Path,
        config_root: Path,
        project_root: Path,
    ) -> None:
        """
        Initialize the config tester.
        
        Args:
            config_dir_from_config_root: Relative path from config root to test dir
            config_root: Absolute path to the config root directory
            project_root: Absolute path to the project root directory
        """
        self.config_root = config_root
        self.project_root = project_root
        self.search_dir_path = config_root / config_dir_from_config_root

    @abstractmethod
    def _init_config(self, rel_config_path: Path, config_name: str) -> DictConfig:
        """Initialize a Hydra config. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _build_base_cmd(self, rel_config_path: Path, config_name: str) -> List[str]:
        """Build the base command to run a config. Must be implemented by subclasses."""
        pass

    def init_config(self, rel_config_path: Path, config_name: str) -> DictConfig:
        """
        Initialize a config, clearing Hydra state first to avoid re-initialization errors.
        
        Args:
            rel_config_path: Relative path from search_dir_path to config directory
            config_name: Name of the config file (without .yaml extension)
            
        Returns:
            The initialized DictConfig
        """
        # Always clear to avoid re-initialisation errors when composing repeatedly
        GlobalHydra.instance().clear()
        return self._init_config(rel_config_path, config_name)

    def test_config(
        self,
        rel_config_path: Path,
        config_name: str,
        timeout: int = DEFAULT_TIMEOUT,
        remove_keys: Optional[List[str]] = None,
        extra_overrides: Optional[List[str]] = None,
        common_overrides: Optional[List[str]] = None,
    ) -> Tuple[str, float]:
        """
        Test a config by running it in a subprocess.
        
        Args:
            rel_config_path: Relative path from search_dir_path to config directory
            config_name: Name of the config file (without .yaml extension)
            timeout: Maximum time to wait for the test to complete (seconds)
            remove_keys: List of config keys to remove with ~ override
            extra_overrides: Additional Hydra overrides to apply
            common_overrides: Common overrides (defaults to BASE_OVERRIDES if None)
            
        Returns:
            Tuple of (stdout, execution_time)
            
        Raises:
            pytest.fail: If the config execution fails
        """
        remove_keys = remove_keys or []
        extra_overrides = extra_overrides or []
        common_overrides = common_overrides if common_overrides is not None else BASE_OVERRIDES.copy()
        
        # Build the cmd to run the experiment
        cmd = self._build_base_cmd(rel_config_path, config_name)

        # Determine which keys to remove
        cfg = self.init_config(rel_config_path, config_name)
        mask = [cfg.get(key, None) not in (None, "", False) for key in remove_keys]
        remove_overrides = [f"~{key}" for key, present in zip(remove_keys, mask) if present]
        
        # Filter out duplicates between common, extra, and remove overrides
        filtered_common = [
            override for override in common_overrides
            if override not in extra_overrides and override not in remove_overrides
        ]
        
        cmd.extend(filtered_common)
        cmd.extend(remove_overrides)
        cmd.extend(extra_overrides)

        # Run the experiment in a subprocess
        start_time = time()
        env = os.environ.copy()
        env.setdefault("HYDRA_FULL_ERROR", "1")

        process = subprocess.run(
            cmd,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        end_time = time()

        execution_time = end_time - start_time
        execution_success = process.returncode == 0
        std_err = process.stderr
        std_out = process.stdout
        
        if not execution_success:
            pytest.fail(
                f"Config '{str(self.search_dir_path / rel_config_path / config_name)}' failed.\n"
                f"Execution time: {execution_time:.2f} seconds\n"
                f"Stdout: {std_out}\n"
                f"Stderr: {std_err}\n"
            )

        return std_out, execution_time
