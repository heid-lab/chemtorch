from collections import OrderedDict
import importlib
import inspect
import os
import random

import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
            ":4096:8"  # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
        )

def check_early_stopping(
    current_loss, best_loss, counter, patience, min_delta
):
    if current_loss < best_loss - min_delta:
        return 0, False
    else:
        counter += 1
        if counter >= patience:
            return counter, True
        return counter, False


def save_model(model, optimizer, epoch, best_val_loss, model_dir):
    """Save model and optimizer state to the model directory."""
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pt")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        },
        model_path,
    )


def load_model(model, optimizer, model_dir):
    """Load model and optimizer state from the model directory."""

    if os.path.exists(model_dir):
        model_path = os.path.join(model_dir, "model.pt")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["best_val_loss"]

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return model, optimizer, epoch, best_val_loss
    else:
        return model, optimizer, 0, float("inf")


def save_standardizer(mean, std, model_dir):
    """Save standardizer parameters to the model directory."""
    os.makedirs(model_dir, exist_ok=True)
    standardizer_path = os.path.join(model_dir, "standardizer.pt")
    torch.save({"mean": mean, "std": std}, standardizer_path)


def load_standardizer(model_dir):
    """Load standardizer parameters from the model directory."""
    standardizer_path = os.path.join(model_dir, "standardizer.pt")
    if os.path.exists(standardizer_path):
        params = torch.load(standardizer_path)
        return params["mean"], params["std"]
    return None, None


def get_generator(seed: int) -> torch.Generator:
    """
    Get a random generator with a specific seed.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


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
        except Exception as e:
            # If target can't be resolved, fallback to original order
            pass
    # Otherwise, just recurse into children
    return {k: order_config_by_signature(v) for k, v in cfg.items()}


def resolve_target(target_str):
    """Resolve a string like 'module.submodule.ClassName' to the actual class."""
    import importlib

    module_path, class_name = target_str.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
