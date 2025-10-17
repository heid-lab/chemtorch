"""Utilities for comparing predictions with reference data."""

from pathlib import Path
from typing import Set, List, Tuple, Optional

import numpy as np
import pandas as pd


def compare_predictions(
    test_preds_path: Path,
    reference_preds_path: Path,
    tolerance: float = 1e-5,
    debug_dir: Optional[Path] = None,
) -> None:
    """
    Compare test predictions with reference predictions.
    
    Args:
        test_preds_path: Path to the test predictions CSV
        reference_preds_path: Path to the reference predictions CSV
        tolerance: Maximum allowed absolute difference
        debug_dir: Optional directory to save debug CSV when errors occur
        
    Raises:
        AssertionError: If predictions don't match within tolerance
    """
    # Load predictions
    test_df = pd.read_csv(test_preds_path)
    ref_df = pd.read_csv(reference_preds_path)
    
    # Basic validation
    assert len(test_df) == len(ref_df), (
        f"Expected {len(ref_df)} predictions, but got {len(test_df)}"
    )
    
    # Check if prediction columns exist
    pred_cols = [col for col in test_df.columns if 'pred' in col.lower()]
    pred_cols_ref = [col for col in ref_df.columns if 'pred' in col.lower()]
    assert pred_cols, "No prediction columns found"
    
    # Track all errors across all prediction columns
    all_error_lines = set()
    error_details = []
    
    for col in pred_cols_ref:
        assert col in pred_cols, (
            f"Prediction column '{col}' not found in test predictions"
        )
        
        errors, details = _compare_column(
            test_df[col].values,
            ref_df[col].values,
            col,
            tolerance,
        )
        all_error_lines.update(errors)
        error_details.extend(details)
    
    # If errors were found, save debug CSV and raise detailed assertion
    if all_error_lines:
        _handle_prediction_errors(
            all_error_lines,
            error_details,
            test_df,
            ref_df,
            pred_cols_ref,
            reference_preds_path,
            debug_dir,
        )


def _compare_column(
    test_values: np.ndarray,
    ref_values: np.ndarray,
    col_name: str,
    tolerance: float,
) -> Tuple[Set[int], List[str]]:
    """
    Compare a single prediction column.
    
    Returns:
        Tuple of (error_line_numbers, error_details)
    """
    test_values = np.array(test_values)
    ref_values = np.array(ref_values)
    
    error_lines = set()
    error_details = []
    
    # Check for NaN mismatches position-wise
    test_is_nan = np.isnan(test_values)
    ref_is_nan = np.isnan(ref_values)
    nan_mismatch_mask = test_is_nan != ref_is_nan
    
    if np.any(nan_mismatch_mask):
        nan_mismatch_indices = np.where(nan_mismatch_mask)[0]
        nan_mismatch_lines = nan_mismatch_indices + 2  # +1 for 0-index, +1 for header
        error_lines.update(nan_mismatch_lines)
        
        for idx, line in zip(nan_mismatch_indices, nan_mismatch_lines):
            error_details.append(
                f"  Line {line}: NaN mismatch in '{col_name}' "
                f"(test={'NaN' if test_is_nan[idx] else test_values[idx]}, "
                f"ref={'NaN' if ref_is_nan[idx] else ref_values[idx]})"
            )
    
    # Check for value differences in non-NaN positions
    valid_mask = ~(test_is_nan | ref_is_nan)
    if np.any(valid_mask):
        differences = np.abs(test_values[valid_mask] - ref_values[valid_mask])
        exceeds_tolerance = differences > tolerance
        
        if np.any(exceeds_tolerance):
            # Get original indices where tolerance is exceeded
            valid_indices = np.where(valid_mask)[0]
            error_indices = valid_indices[exceeds_tolerance]
            error_line_nums = error_indices + 2  # +1 for 0-index, +1 for header
            error_lines.update(error_line_nums)
            
            for idx, line in zip(error_indices, error_line_nums):
                diff = abs(test_values[idx] - ref_values[idx])
                error_details.append(
                    f"  Line {line}: Value exceeds tolerance in '{col_name}' "
                    f"(test={test_values[idx]:.10f}, ref={ref_values[idx]:.10f}, "
                    f"diff={diff:.2e})"
                )
    
    return error_lines, error_details


def _handle_prediction_errors(
    error_lines: Set[int],
    error_details: List[str],
    test_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    pred_cols: List[str],
    reference_path: Path,
    debug_dir: Optional[Path] = None,
) -> None:
    """Save debug CSV and raise detailed error."""
    # Create debug dataframe
    debug_df = ref_df.copy()
    
    # Add test prediction columns
    for col in pred_cols:
        test_col_name = f"test_{col}"
        debug_df[test_col_name] = test_df[col]
    
    # Add error flag column
    debug_df['has_error'] = False
    error_indices = [line - 2 for line in error_lines]  # Convert back to 0-indexed
    debug_df.loc[error_indices, 'has_error'] = True
    
    # Save debug CSV if directory provided
    debug_path = None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_path = debug_dir / reference_path.name
        debug_df.to_csv(debug_path, index=False)
    
    # Build error message
    sorted_error_lines = sorted(error_lines)
    error_msg = f"Found {len(sorted_error_lines)} lines with prediction mismatches:\n"
    error_msg += "\n".join(error_details)
    
    if debug_path:
        error_msg += f"\n\nDebug CSV saved to: {debug_path}"
    error_msg += f"\nError lines: {sorted_error_lines}"
    
    raise AssertionError(error_msg)
