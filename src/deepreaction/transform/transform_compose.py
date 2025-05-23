from typing import TypeVar, Sequence

from deepreaction.transform.abstract_transform import AbstractTransform

T = TypeVar("T")

class TransformCompose(AbstractTransform[T]):
    def __init__(self, transforms: Sequence[AbstractTransform[T]]):
        self.transforms = transforms

    def __call__(self, data: T) -> T:
        for t in self.transforms:
            data = t(data)
        return data