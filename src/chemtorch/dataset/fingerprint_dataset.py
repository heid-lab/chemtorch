import torch
from torch.utils.data import Dataset

from chemtorch.dataset import DatasetBase

class FingerprintDataset(DatasetBase[torch.Tensor], Dataset):

    @property
    def fp_length(self) -> int:
        """Returns the length of fingerprints in the dataset."""
        data = self[0]
        fingerprint = data[0]   # data is a tuple (fingerprint, label)

        if not isinstance(fingerprint, torch.Tensor):
            raise AttributeError(
                f"'{fingerprint.__class__.__name__}' object cannot be used "
                f"to determine fingerprint length"
            )
        
        return fingerprint.shape[0]