from typing import List, Optional
import torch
from torch._tensor import Tensor

from chemtorch.components.transform.graph_transform.ts_3d_jitter import TS3DJitterTransform

try:
    # Python ≥ 3.12
    from typing import override  # type: ignore
except ImportError:
    # Python < 3.12
    from typing_extensions import override  # type: ignore

from chemtorch.components.augmentation.abstract_augmentation import AbstractAugmentation
from chemtorch.components.representation.graph.reaction_3d_graph import Reaction3DData


class TS3DJitterAugmentation(AbstractAugmentation[Reaction3DData]):
    """
    Apply Gaussian noise to the 3D coordinates of the transition state.
    """

    def __init__(self, jitter: TS3DJitterTransform, num_augmentations: int = 1):
        self.jitter = jitter
        self.num_augmentations = num_augmentations

    @override
    def _augment(self, obj: Reaction3DData, label: Optional[Tensor] = None) -> List[tuple[Reaction3DData, Optional[Tensor]]]:
        augmented_list = []

        for _ in range(self.num_augmentations):
            augmented_data = self.jitter(obj.clone())
            augmented_list.append((augmented_data, label))

        return augmented_list
