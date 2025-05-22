from typing import TypeVar, Sequence

from deepreaction.transform.abstract_transform import AbstractTransform

T = TypeVar("T")

class Compose(AbstractTransform[T]):
    def __init__(self, transforms: Sequence[AbstractTransform[T]]):
        self.transforms = transforms

    def forward(self, data: T) -> T:
        for t in self.transforms:
            data = t(data)
        return data