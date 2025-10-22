from collections import OrderedDict
import os
import sys

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import inspect
from hydra._internal.instantiate._instantiate2 import _Keys  # Import the enum


def resolve_target(target_str):
    """Resolve a string like 'module.submodule.ClassName' to the actual class."""
    import importlib

    module_path, class_name = target_str.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def order_config_by_signature(cfg):
    """
    Recursively reorder configuration dictionaries to match the argument order of their
    specified `_target_` class or function's constructor signature.

    Args:
        cfg (dict or DictConfig): The configuration dictionary or OmegaConf DictConfig.

    Returns:
        dict: A reordered configuration dictionary with keys ordered according to the
              constructor signature of the `_target_` if present, otherwise recursively
              processes child dictionaries.

    Why is this important?
    ----------------------
    By default, Hydra instantiates objects in the order that keys appear in the configuration
    dictionary. However, the order of keys in a YAML or Python dict is not guaranteed to match
    the order of arguments in the target class or function's constructor. In frameworks like
    PyTorch, the order in which submodules (layers, blocks, etc.) are instantiated directly
    affects the order in which random numbers are consumed for parameter initialization.
    This means that simply reordering keys in the config file can lead to different random 
    initializations and thus different results, even if the model architecture and random seed
    remain unchanged.

    This function ensures that the instantiation order of all objects (and their subcomponents)
    is invariant to the order of keys in the configuration file. It does so by reordering
    each config dictionary to match the argument order of the corresponding `_target_`s
    constructor. This guarantees reproducible model initialization and results, regardless
    of how the config is written.
    """
    if isinstance(cfg, DictConfig):
        # Do NOT resolve interpolations here!
        cfg = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(cfg, dict):
        return cfg
    # If this node has a _target_, reorder its keys
    if "_target_" in cfg:
        try:
            target = resolve_target(cfg["_target_"])
            if hasattr(target, "__init__"):
                sig = inspect.signature(target.__init__)
            else:
                sig = inspect.signature(target)
            param_names = [
                p.name
                for p in sig.parameters.values()
                if p.name != "self" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            ]
            ordered = OrderedDict()
            for name in param_names:
                if name in cfg:
                    ordered[name] = order_config_by_signature(cfg[name])
            # Add any extra keys (including _target_)
            for k in cfg:
                if k not in ordered:
                    ordered[k] = order_config_by_signature(cfg[k])
            return dict(ordered)  # <-- convert to dict before returning!
        except Exception:
            # If target can't be resolved, fallback to original order
            pass
    # Otherwise, just recurse into children
    return {k: order_config_by_signature(v) for k, v in cfg.items()}


def filter_config_by_signature(cfg):
    """
    Recursively filter configuration dictionaries to only include keys that match the argument
    names of their specified `_target_` class or function's constructor signature.
    If the signature supports **kwargs, do not filter.
    Special Hydra keys (from _Keys) are always preserved.

    Args:
        cfg (dict or DictConfig): The configuration dictionary or OmegaConf DictConfig.

    Returns:
        dict: A filtered configuration dictionary.

    Why is this important?
    ----------------------
    This is useful for convenient key interpolation in configs. If multiple lower-level configs
    in the hierarchy depend on a shared value, it is convenient to define it in a higher-level
    config and use a universal interpolation syntax to reference that key, regardless of the
    lower-level config structure (which can differ). For example, the `hidden_channels` argument
    is shared across multiple components of a GNN, such as encoders, convolution layers, and
    higher-level blocks. However, the higher-level GNN class that assembles these components
    might not have a `hidden_channels` argument itself. If you specify a `hidden_channels` key
    in the higher-level config and use top-level interpolation, Hydra will raise a ValueError
    when it tries to instantiate the GNN with the config, since the argument is unexpected.
    The use case for this function is to resolve the config (so all interpolations are applied),
    then filter it so that only valid keys for each `_target_` remain, avoiding instantiation
    errors and enabling flexible, DRY config design.
    """
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(cfg, dict):
        return cfg

    hydra_keys = {k.value for k in _Keys}  # Set of special hydra keys

    if "_target_" in cfg:
        try:
            target = resolve_target(cfg["_target_"])
            if hasattr(target, "__init__"):
                sig = inspect.signature(target.__init__)
            else:
                sig = inspect.signature(target)
            # If the signature accepts **kwargs, do not filter
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                return {k: filter_config_by_signature(v) for k, v in cfg.items()}
            param_names = {
                p.name
                for p in sig.parameters.values()
                if p.name != "self" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            }
            filtered = {}
            for k, v in cfg.items():
                if k in hydra_keys or k in param_names:
                    filtered[k] = filter_config_by_signature(v)
            return filtered
        except Exception:
            # If target can't be resolved, fallback to original
            pass
    # Otherwise, just recurse into children
    return {k: filter_config_by_signature(v) for k, v in cfg.items()}

def safe_instantiate(cfg, *args, **kwargs):
    """
    Safely instantiate an object from a configuration dictionary, ensuring that
    the configuration is ordered and filtered according to the target's constructor signature.

    Args:
        cfg (dict or DictConfig): The configuration dictionary or OmegaConf DictConfig.
        *args: Positional arguments to pass to the target's constructor.
        **kwargs: Keyword arguments to pass to the target's constructor.

    Returns:
        object: An instance of the target class specified in the configuration.
    """
    cfg = order_config_by_signature(cfg)
    cfg = filter_config_by_signature(cfg)
    return instantiate(cfg, *args, **kwargs)

def get_num_workers(slurm_env: str = 'SLURM_CPUS_PER_TASK', leave_free: int = 1) -> int:
    """Return a safe number of PyTorch DataLoader workers.

    Behavior:
    - If the SLURM env var (default: ``SLURM_CPUS_PER_TASK``) is present and an
      integer, use min(slurm_value, os.cpu_count()) - leave_free.
    - Otherwise use os.cpu_count() - leave_free.
    - Never return a negative number; result is clamped to 0.

    leave_free defaults to 1 to leave one CPU for the main process. This is a
    common pattern but optional depending on your workload.
    """
    # On macOS (Darwin) the default multiprocessing start method is 'spawn',
    # which requires that objects passed to worker processes be picklable.
    # Many C/C++ extension callables (e.g. Boost.Python function objects) are
    # not picklable and will raise a TypeError inside workers. To avoid
    # surprising crashes for mac users, default to 0 workers on Darwin unless
    # explicitly overridden by the env var ``CHEMTORCH_ALLOW_MULTIPROC_ON_MAC``.
    if sys.platform == 'darwin':
        allow = os.getenv('CHEMTORCH_ALLOW_MULTIPROC_ON_MAC', '')
        if allow not in ('1', 'true', 'True'):
            return 0

    slurm_val = os.getenv(slurm_env)
    slurm = None
    if slurm_val not in (None, ''):
        try:
            slurm = int(slurm_val)
        except (ValueError, TypeError):
            # Ignore invalid SLURM value and fall back to cpu_count()
            slurm = None

    cpu_count = os.cpu_count() or 1

    if slurm is not None:
        workers = min(slurm, cpu_count) - leave_free
    else:
        workers = cpu_count - leave_free

    return max(0, int(workers))