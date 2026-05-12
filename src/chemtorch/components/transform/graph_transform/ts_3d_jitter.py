import torch

try:
    # Python ≥ 3.12
    from typing import override  # type: ignore
except ImportError:
    # Python < 3.12
    from typing_extensions import override  # type: ignore

from chemtorch.components.representation.graph.reaction_3d_graph import Reaction3DData
from chemtorch.components.transform.abstract_transform import AbstractTransform


class TS3DJitterTransform(AbstractTransform[Reaction3DData]):
    """
    Apply Gaussian noise to the 3D coordinates of the transition state.

    The noise is sampled from a normal distribution with mean 0 and standard deviation defined by `sigma`.
    The noise is clipped to the range [-clip, clip] to prevent excessive perturbations.
    """

    def __init__(self, sigma: float = 0.1, clip: float = 0.5):
        self.sigma = sigma
        self.clip = clip

    def _make_jitter(self, pos: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            torch.randn_like(pos) * self.sigma, -self.clip, self.clip
        )

    @override
    def __call__(self, obj: Reaction3DData, **kwargs) -> Reaction3DData:
        def _make_jitter(pos: torch.Tensor) -> torch.Tensor:
            return torch.clamp(
                torch.randn_like(pos) * self.sigma, -self.clip, self.clip
            )

        ts_jitter = _make_jitter(obj.pos_ts)

        obj.pos_ts += ts_jitter

        return obj