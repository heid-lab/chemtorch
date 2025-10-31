"""Utilities for extracting metrics from test outputs."""

from typing import Optional


def extract_val_loss(output: str) -> Optional[float]:
    """
    Extract validation loss from CLI output.
    
    Args:
        output: The stdout from running a config test
        
    Returns:
        The validation loss value from the last epoch, or None if not found
    """
    lines = output.split('\n')
    
    # Look for Lightning progress bar output with val_loss_epoch
    # Iterate from the end to find the LAST occurrence (final epoch's loss)
    for line in reversed(lines):
        # Look for validation metrics in progress bar: "val_loss_epoch=1.280"
        if "val_loss_epoch=" in line:
            try:
                # Extract val_loss_epoch value from progress bar
                parts = line.split("val_loss_epoch=")
                if len(parts) > 1:
                    val_part = parts[1].split(",")[0].split("]")[0].strip()
                    return float(val_part)
            except (ValueError, IndexError):
                continue
    
    return None
