from typing import Optional
from tap import Tap
import random
import numpy as np
import torch

class TrainArgs(Tap):
    seed: Optional[int] = None
    """Random seed to use for data splitting and model initialization."""
    data_source: str = "online"
    """Source of the data. Can be 'online' or one of the local folder names."""

    def process_args(self) -> None:
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
