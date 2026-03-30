from numbers import Number
from typing import Any, Tuple

import numpy as np
import scipy.special
import torch


class IveFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, v: Number, z: torch.Tensor) -> torch.Tensor:
        assert isinstance(v, Number), "v must be a scalar"

        ctx.save_for_backward(z)
        ctx.v = v
        z_cpu = z.double().detach().cpu().numpy()

        if np.isclose(v, 0):
            output = scipy.special.i0e(z_cpu, dtype=z_cpu.dtype)
        elif np.isclose(v, 1):
            output = scipy.special.i1e(z_cpu, dtype=z_cpu.dtype)
        else:
            output = scipy.special.ive(v, z_cpu, dtype=z_cpu.dtype)

        return torch.tensor(output, dtype=z.dtype, device=z.device)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[None, torch.Tensor]:
        z = ctx.saved_tensors[-1]
        return None, grad_output * (ive(ctx.v - 1, z) - ive(ctx.v, z) * (ctx.v + z) / z)


def ive(v: Number, z: torch.Tensor) -> torch.Tensor:
    return IveFunction.apply(v, z)
