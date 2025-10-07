import logging
from pathlib import Path
from typing import Optional, List, Union, Callable, Any

from omegaconf import ListConfig
import pandas as pd
import torch


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
    logging.info(f"Predictions saved to: {output_path}")
    
    # Log results if logging function provided
    if log_func:
        log_func({
            "predictions_file": str(output_path),
            "num_predictions": len(pred_values)
        }, commit=False)

def handle_prediction_saving(
    get_preds_func: Callable[[str], List[Any]],
    get_reference_df_func: Callable[[str], pd.DataFrame],
    get_dataset_names_func: Callable[[], List[str]],
    predictions_save_dir: Optional[str] = None,
    predictions_save_path: Optional[str] = None,
    save_predictions_for: Optional[Union[str, List[str], ListConfig]] = None,
    tasks: Optional[Union[List[str], ListConfig]] = None,
    log_func: Optional[Callable] = None,
    root_dir: Optional[Path] = None
) -> None:
    """
    Handle prediction saving logic with configurable prediction and reference data retrieval.

    Args:
        get_preds_func: Function that takes a dataset_key and returns predictions
        get_reference_df_func: Function that takes a dataset_key and returns reference dataframe
        get_dataset_names_func: Function that returns all available dataset names/keys
        predictions_save_dir: Directory to save predictions (for multiple partitions)
        predictions_save_path: Specific path to save predictions (for single partition)
        save_predictions_for: String or list of dataset keys to save predictions for
        tasks: List of tasks being executed (used for determining available partitions)
        log_func: Optional logging function (e.g., wandb.log)
        root_dir: Root directory for relative path resolution
    """
    # Prediction saving logic with closures
    _save_predictions = predictions_save_dir or predictions_save_path

    if not _save_predictions and save_predictions_for:
        logging.warning("'save_predictions_for' specified but no 'predictions_save_dir' or 'predictions_save_path' given. Skipping prediction saving.")
    
    if _save_predictions:
        # 1. Determine save_predictions_for list 
        save_predictions_for_list = []

        if save_predictions_for:
            # Normalize save_predictions_for to always be a list if provided
            if isinstance(save_predictions_for, str):
                # Handle special case where Hydra passes a list-like string (e.g., "[test]" -> ["test"])
                if save_predictions_for.startswith('[') and save_predictions_for.endswith(']'):
                    # Remove brackets and split by comma, then strip whitespace
                    inner_content = save_predictions_for[1:-1].strip()
                    if inner_content:
                        normalized_save_predictions_for = [item.strip() for item in inner_content.split(',')]
                    else:
                        normalized_save_predictions_for = []
                else:
                    normalized_save_predictions_for = [save_predictions_for]
            elif isinstance(save_predictions_for, (list, ListConfig)):
                normalized_save_predictions_for = list(save_predictions_for)
            else:
                raise ValueError("'save_predictions_for' must be a string or list of strings if provided.")

            # Validate save_predictions_for contents
            allowed_keys = {"train", "val", "test", "predict", "all"}
            invalid_keys = [key for key in normalized_save_predictions_for if key not in allowed_keys]
            if invalid_keys:
                raise ValueError(
                    f"Invalid entries in 'save_predictions_for': {invalid_keys}. "
                    f"Must be one of {sorted(allowed_keys)}."
                )

            if "all" in normalized_save_predictions_for:
                if len(normalized_save_predictions_for) > 1:
                    logging.warning("Found 'all' in save_predictions_for list, saving predictions for all partitions.")
                
                # only add those that are available for the given tasks list
                available_partitions = set()
                if tasks and "fit" in tasks:
                    available_partitions.update(["train", "val"])
                if tasks and "validate" in tasks:
                    available_partitions.add("val")
                if tasks and "test" in tasks:
                    available_partitions.add("test")
                if tasks and "predict" in tasks:
                    available_partitions.add("predict")

                normalized_save_predictions_for = list(available_partitions)

        # For single partition scenarios (predict, validate, test), use predictions_save_path
        is_single_task = tasks and "fit" not in tasks and len(tasks) == 1
        is_single_partition = (
            save_predictions_for
            and len(normalized_save_predictions_for) == 1
            and "all" not in (
                save_predictions_for
                if isinstance(save_predictions_for, list)
                else [save_predictions_for]
            )
        )
        if is_single_task or is_single_partition:
            if save_predictions_for:
                dataset_key = normalized_save_predictions_for[0] if normalized_save_predictions_for[0] != "validate" else "val"
                save_predictions_for_list = [dataset_key]
            else:
                task = tasks[0] if tasks else "predict"
                dataset_key = task if task != "validate" else "val"    # map to 'validate' task to 'val' dataset key
                save_predictions_for_list = [dataset_key]

        # For multi-partition scenarios (fit or multiple tasks or 'all')
        else:
            if not predictions_save_dir or not save_predictions_for:
                logging.warning(
                    "'save_predictions_for' has to be set to save predictions when using multiple data partitions (i.e., when tasks include 'fit' or multiple tasks)."
                )
                return

            save_predictions_for_list = normalized_save_predictions_for
        
        # 2. Generate and save predictions for each specified partition
        available_dataset_names = get_dataset_names_func()
        
        for dataset_key in available_dataset_names:
            # Check if this dataset should be saved based on save_predictions_for_list
            should_save = False
            for save_key in save_predictions_for_list:
                if save_key == dataset_key or dataset_key.startswith(f"{save_key}_"):
                    should_save = True
                    break
            
            if should_save:
                # Use the closures to get predictions and reference dataframe
                preds = get_preds_func(dataset_key)
                pred_df = get_reference_df_func(dataset_key)

                if preds:
                    if len(save_predictions_for_list) == 1 and predictions_save_path:
                        # Use user-specified path directly for single partition scenarios
                        save_path = predictions_save_path
                    else:
                        save_path = f"{predictions_save_dir}/{dataset_key}_preds.csv"

                    save_predictions(
                        preds=preds,
                        reference_df=pred_df,
                        save_path=save_path,
                        log_func=log_func,
                        root_dir=root_dir,
                    )
