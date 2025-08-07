import torch
from torch.utils.data import Dataset

from chemtorch.components.dataset.dataset_base import DatasetBase

class TokenDataset(DatasetBase[torch.Tensor], Dataset):
    """
    A dataset for string-based tokens.

    It supports in-memory precomputation, caching, and
    subsampling for efficient data handling.
    """
    
    @property
    def vocab_size(self) -> int:
        return len(self.representation.word2id)
