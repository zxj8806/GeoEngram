from typing import Any, Tuple, Callable

import torch
from torch import Tensor


class Manifold:

    def sample_projection_mu0(self, xexpo: Tensor, at_point: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        raise NotImplementedError

    def inverse_sample_projection_mu0(self, x_proj: Tensor, at_point: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def logdet(self, mu: Tensor, std: Tensor, z: Tensor, data: Tuple[Tensor, ...]) -> Tensor:
        raise NotImplementedError

    @property
    def radius(self) -> Tensor:
        raise NotImplementedError

    @property
    def curvature(self) -> Tensor:
        raise NotImplementedError

    def exp_map_mu0(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def inverse_exp_map_mu0(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def parallel_transport_mu0(self, x: Tensor, dst: Tensor) -> Tensor:
        raise NotImplementedError

    def inverse_parallel_transport_mu0(self, x: Tensor, src: Tensor) -> Tensor:
        raise NotImplementedError

    def mu_0(self, shape: torch.Size, **kwargs: Any) -> Tensor:
        raise NotImplementedError


class RadiusManifold(Manifold):

    def __init__(self, radius: Callable[[], Tensor]):
        super().__init__()
        self._radius = radius

    @property
    def curvature(self) -> Tensor:
        return 1. / self.radius.pow(2)

    @property
    def radius(self) -> Tensor:
        return torch.clamp(torch.relu(self._radius()), min=1e-8, max=1e8)
