from typing import Any, Tuple
from geoopt import PoincareBall
manifold = PoincareBall()

import geoopt.manifolds.stereographic.math as pm
import torch
import torch.distributions
from torch import Tensor

from .manifold import RadiusManifold
from .common import eps, sqrt, atanh
from . import hyperbolics as H


class PoincareBall(RadiusManifold):

    @property
    def curvature(self) -> Tensor:
        return -super().curvature

    def exp_map_mu0(self, x: Tensor) -> Tensor:
        return exp_map_mu0(x, radius=self.radius)

    def inverse_exp_map_mu0(self, x: Tensor) -> Tensor:
        return inverse_exp_map_mu0(x, radius=self.radius)

    def parallel_transport_mu0(self, x: Tensor, dst: Tensor) -> Tensor:
        return parallel_transport_mu0(x, dst, radius=self.radius)

    def inverse_parallel_transport_mu0(self, x: Tensor, src: Tensor) -> Tensor:
        return inverse_parallel_transport_mu0(x, src, radius=self.radius)

    def mu_0(self, shape: torch.Size, **kwargs: Any) -> Tensor:
        return mu_0(shape, **kwargs)

    def sample_projection_mu0(self, x: Tensor, at_point: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        return sample_projection_mu0(x, at_point, radius=self.radius)

    def inverse_sample_projection_mu0(self, x_proj: Tensor, at_point: Tensor) -> Tuple[Tensor, Tensor]:
        return inverse_sample_projection_mu0(x_proj, at_point, radius=self.radius)

    def logdet(self, mu: Tensor, std: Tensor, z: Tensor, data: Tuple[Tensor, ...]) -> Tensor:
        b_mu = mu
        if len(z.shape) > 2:
            b_mu = mu.unsqueeze(dim=0).repeat((z.shape[0], 1, 1))

        z_sphere = poincare_to_lorentz(z, self.radius)
        mu_sphere = poincare_to_lorentz(b_mu, self.radius)
        u, v = H.inverse_sample_projection_mu0(z_sphere, mu_sphere, radius=self.radius)
        logdet_partial = H._logdet(u, self.radius)
        assert torch.isfinite(logdet_partial).all()
        return logdet_partial


def poincare_distance(x: Tensor, y: Tensor, radius: Tensor, **kwargs: Any) -> Tensor:
    return poincare_distance_c(x, y, _c(radius), **kwargs)


def poincare_distance_c(x: Tensor, y: Tensor, c: Tensor, keepdim: bool = True, **kwargs: Any) -> Tensor:
    sqrt_c = sqrt(c)
    mob = pm.mobius_add(-x, y, k=-c, dim=-1).norm(dim=-1, p=2, keepdim=keepdim)
    arg = sqrt_c * mob
    dist_c = atanh(arg)
    res = dist_c * 2 / sqrt_c
    assert torch.isfinite(res).all()
    return res


def _c(radius: Tensor) -> Tensor:
    return 1 / radius**2


def mu_0(shape: Tuple[int, ...], **kwargs: Any) -> Tensor:
    return torch.zeros(shape, **kwargs)


def parallel_transport_mu0(x: Tensor, dst: Tensor, radius: Tensor) -> Tensor:
    return pm.parallel_transport0(dst, x, k=-_c(radius), dim=-1)


def inverse_parallel_transport_mu0(x: Tensor, src: Tensor, radius: Tensor) -> Tensor:
    return pm.parallel_transport0back(src, x, k=-_c(radius), dim=-1)


def exp_map(x: Tensor, at_point: Tensor, radius: Tensor) -> Tensor:
    return exp_map_c(x, at_point, c=_c(radius))


def exp_map_c(x: Tensor, at_point: Tensor, c: Tensor) -> Tensor:
    return pm.expmap(at_point, x, k=-c, dim=-1)


def exp_map_mu0(x: Tensor, radius: Tensor) -> Tensor:
    return exp_map_mu0_c(x, c=_c(radius))


def exp_map_mu0_c(x: Tensor, c: Tensor) -> Tensor:
    return pm.expmap0(x, k=-c, dim=-1)


def inverse_exp_map(x: Tensor, at_point: Tensor, radius: Tensor) -> Tensor:
    return inverse_exp_map_c(x, at_point, c=_c(radius))


def inverse_exp_map_c(x: Tensor, at_point: Tensor, c: Tensor) -> Tensor:
    return pm.logmap(at_point, x, k=-c, dim=-1)


def inverse_exp_map_mu0(x: Tensor, radius: Tensor) -> Tensor:
    return pm.logmap0(x, k=-_c(radius), dim=-1)


def sample_projection_mu0(x: Tensor, at_point: Tensor, radius: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
    c = _c(radius)
    v_ = x / pm.lambda_x(at_point, k=-c, dim=-1, keepdim=True)
    x_proj = exp_map(v_, at_point=at_point, radius=radius)
    return x_proj, (v_, x)


def inverse_sample_projection_mu0(x_proj: Tensor, at_point: Tensor, radius: Tensor) -> Tuple[Tensor, Tensor]:
    c = _c(radius)
    v_ = inverse_exp_map_c(x_proj, at_point=at_point, c=c)
    x = v_ * pm.lambda_x(at_point, k=-c, dim=-1, keepdim=True)
    return v_, x


def poincare_to_lorentz(y: Tensor, radius: Tensor) -> Tensor:
    norm = torch.norm(y, p=2, dim=-1, keepdim=True)**2
    denom = radius**2 - norm
    return torch.cat((radius * (radius**2 + norm), 2 * radius**2 * y), dim=-1) / denom


def exp_map_x_polar(x: Tensor, radius: Tensor, v: Tensor, c: Tensor) -> Tensor:
    v = v + eps
    norm_v = torch.norm(v, p=2, dim=-1, keepdim=True)
    assert len(radius.shape) == len(v.shape)
    assert radius.shape[:-1] == v.shape[:-1]

    second_term = (torch.tanh(c.sqrt() * radius / 2) / (c.sqrt() * norm_v)) * v
    assert x.shape == second_term.shape
    return pm.mobius_add(x, second_term, k=-c, dim=-1)