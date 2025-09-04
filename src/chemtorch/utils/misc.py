import os
import random
from pathlib import Path
from typing import Optional, List, Union, Callable, Any

import numpy as np
import pandas as pd
import torch



def save_standardizer(mean, std, model_dir):
    """DEPRECATED: Save standardizer parameters to the model directory."""
    os.makedirs(model_dir, exist_ok=True)
    standardizer_path = os.path.join(model_dir, "standardizer.pt")
    torch.save({"mean": mean, "std": std}, standardizer_path)


def load_standardizer(model_dir):
    """DEPRECATED. Load standardizer parameters from the model directory."""
    standardizer_path = os.path.join(model_dir, "standardizer.pt")
    if os.path.exists(standardizer_path):
        params = torch.load(standardizer_path)
        return params["mean"], params["std"]
    return None, None


def get_generator(seed: int) -> torch.Generator:
    """DEPRECATED. Get a random generator with a specific seed."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def save_predictions(
    preds: List[Any],
    reference_df: pd.DataFrame,
    save_path: Optional[Union[str, Path]],
    log_func: Optional[Callable[..., Any]] = None,
    root_dir: Optional[Path] = None
) -> Optional[pd.DataFrame]:
    """
    Process predictions and save them to a dataframe with the original data.
    
    Args:
        preds: List of predictions from trainer.predict() or trainer.test() (can be tensors or other types)
               Each element in the list is typically a batch of predictions
        reference_df: Original dataframe to add predictions to
        save_path: Path to save the predictions CSV file (relative to root_dir if provided)
        log_func: Optional logging function (e.g., wandb.log)
        root_dir: Root directory for resolving relative save_path
        
    Returns:
        DataFrame with predictions added, or None if processing failed
    """
    if preds is None or reference_df is None:
        return
    
    if save_path is None:
        return

    # Convert tensor predictions to numpy values
    # preds is a list of batches, need to flatten all predictions
    pred_values = []
    for batch_pred in preds:
        if isinstance(batch_pred, torch.Tensor):
            # Handle batch of predictions
            if batch_pred.dim() == 0:
                # Single scalar prediction
                pred_values.append(batch_pred.item())
            elif batch_pred.dim() == 1:
                # Batch of scalar predictions
                pred_values.extend(batch_pred.cpu().numpy())
            else:
                # Multi-dimensional predictions, flatten
                pred_values.extend(batch_pred.cpu().numpy().flatten())
        elif isinstance(batch_pred, (list, tuple)):
            # Handle case where batch_pred is a list/tuple of tensors
            for pred in batch_pred:
                if isinstance(pred, torch.Tensor):
                    if pred.numel() == 1:
                        pred_values.append(pred.item())
                    else:
                        pred_values.extend(pred.cpu().numpy().flatten())
                else:
                    pred_values.append(pred)
        else:
            pred_values.append(batch_pred)
    
    # Check if predictions match dataframe size
    if len(pred_values) != len(reference_df):
        raise RuntimeError(f"Number of predictions ({len(pred_values)}) doesn't match dataset size ({len(reference_df)})")
    
    # Create new dataframe with predictions
    result_df = reference_df.copy()
    result_df['prediction'] = pred_values
    
    if root_dir:
        output_path = root_dir / save_path
    else:
        output_path = Path(save_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
    
    # Log results if logging function provided
    if log_func:
        log_func({
            "predictions_file": str(output_path),
            "num_predictions": len(pred_values)
        }, commit=False)

def set_seed(seed):
    """DEPRECATED"""
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
    """DEPRECATED"""
    if current_loss < best_loss - min_delta:
        return 0, False
    else:
        counter += 1
        if counter >= patience:
            return counter, True
        return counter, False


def save_model(model, optimizer, epoch, best_val_loss, model_dir):
    """DEPRECATED. Save model and optimizer state to the model directory."""
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
    """DEPRECATED. Load model and optimizer state from the model directory."""

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
