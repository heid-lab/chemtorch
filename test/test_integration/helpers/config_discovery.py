"""
Utilities for discovering and parametrizing YAML config tests.
"""

from pathlib import Path
from typing import List, Tuple


def discover_yaml_configs(search_dir: Path) -> List[Tuple[Path, str]]:
    """
    Discover all YAML config files in a directory and subdirectories.

    Args:
        search_dir: The directory to search for YAML config files.
    
    Returns:
        A list of tuples containing (relative_path_from_search_dir, config_name).
    """
    configs = []
    
    for yaml_file in search_dir.rglob("*.yaml"):
        config_name = yaml_file.stem
        relative_path = Path(yaml_file.relative_to(search_dir)).parent
        configs.append((relative_path, config_name))
    
    return sorted(configs)


def parametrize_config_tests(metafunc, search_dir: Path, *, argname: str = "config_info") -> None:
    """
    Parametrize a pytest test function with all configs discovered under search_dir.
    
    Args:
        metafunc: The pytest metafunc object.
        search_dir: Directory to search for YAML configs.
        argname: The argument name to parametrize (default: "config_info").
    """
    if argname not in metafunc.fixturenames:
        return

    configs = discover_yaml_configs(search_dir)
    config_ids = [str(path / name) for path, name in configs]
    metafunc.parametrize(argname, configs, ids=config_ids)
