from typing import TypeVar, Sequence

from deepreaction.transform.transform_base import TransformBase

T = TypeVar("T")

class TransformCompose(TransformBase[T]):
    def __init__(self, transforms: Sequence[TransformBase[T]]):
        self.transforms = transforms

    def forward(self, data: T) -> T:
        for t in self.transforms:
            data = t(data)
        return data