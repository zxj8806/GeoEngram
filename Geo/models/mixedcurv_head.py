import torch
import torch.nn as nn
import torch.nn.functional as F

from MixCurv.ops import Euclidean, PoincareBall, StereographicallyProjectedSphere
from MixCurv.ops import poincare as P
from MixCurv.ops import spherical_projected as SP


class MixedCurvProtoHead(nn.Module):

    def __init__(
        self,
        in_dim: int,
        n_classes: int = 10,
        d_e: int = 64,
        d_p: int = 32,
        d_s: int = 32,
        init_radius: float = 1.0,
        learn_radius: bool = True,
        temp: float = 0.1,
        eps: float = 1e-6,
        learn_branch_weights: bool = True,
        route_topk: int = 64,
        branch_warmup_steps: int = 1000,
        branch_floor: float = 0.05,
        ema_momentum: float = 0.99,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.n_classes = int(n_classes)
        self.d_e = int(d_e)
        self.d_p = int(d_p)
        self.d_s = int(d_s)
        self.temp = float(temp)
        self.eps = float(eps)
        self.route_topk = int(route_topk)
        self.branch_warmup_steps = int(branch_warmup_steps)
        self.branch_floor = float(branch_floor)
        self.ema_momentum = float(ema_momentum)

        self.r_p = nn.Parameter(torch.tensor(float(init_radius)), requires_grad=bool(learn_radius))
        self.r_s = nn.Parameter(torch.tensor(float(init_radius)), requires_grad=bool(learn_radius))

        self.M_e = Euclidean()
        self.M_p = PoincareBall(lambda: self.r_p)
        self.M_s = StereographicallyProjectedSphere(lambda: self.r_s)

        self.proj_e = nn.Linear(self.in_dim, self.d_e)
        self.proj_p = nn.Linear(self.in_dim, self.d_p)
        self.proj_s = nn.Linear(self.in_dim, self.d_s)

        self.proto_e_tan = nn.Parameter(torch.randn(self.n_classes, self.d_e) * 0.01)
        self.proto_p_tan = nn.Parameter(torch.randn(self.n_classes, self.d_p) * 0.01)
        self.proto_s_tan = nn.Parameter(torch.randn(self.n_classes, self.d_s) * 0.01)

        self.branch_logits = nn.Parameter(torch.zeros(3), requires_grad=bool(learn_branch_weights))
        self.register_buffer("forward_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("ema_de", torch.ones((), dtype=torch.float32))
        self.register_buffer("ema_dp", torch.ones((), dtype=torch.float32))
        self.register_buffer("ema_ds", torch.ones((), dtype=torch.float32))

    def _branch_weights(self) -> torch.Tensor:
        if not self.branch_logits.requires_grad:
            return torch.full((3,), 1.0 / 3.0, device=self.branch_logits.device, dtype=self.branch_logits.dtype)
        if self.training and int(self.forward_count.item()) < self.branch_warmup_steps:
            return torch.full((3,), 1.0 / 3.0, device=self.branch_logits.device, dtype=self.branch_logits.dtype)
        w = torch.softmax(self.branch_logits, dim=0)
        if self.branch_floor > 0:
            w = (1.0 - self.branch_floor) * w + self.branch_floor / 3.0
        return w / w.sum()

    def _state_signature(self, x: torch.Tensor) -> torch.Tensor:
        k = max(1, min(self.route_topk, x.size(-1)))
        vals, idx = torch.topk(x, k=k, dim=-1)
        sig = torch.zeros_like(x)
        sig.scatter_(dim=-1, index=idx, src=vals)
        sig = F.normalize(sig, p=2, dim=-1)
        return sig

    @torch.no_grad()
    def _update_distance_ema(self, de2: torch.Tensor, dp2: torch.Tensor, ds2: torch.Tensor) -> None:
        m = self.ema_momentum
        self.ema_de.mul_(m).add_((1.0 - m) * de2.detach().mean())
        self.ema_dp.mul_(m).add_((1.0 - m) * dp2.detach().mean())
        self.ema_ds.mul_(m).add_((1.0 - m) * ds2.detach().mean())

    def _normalize_distances(self, de2: torch.Tensor, dp2: torch.Tensor, ds2: torch.Tensor):
        if self.training:
            self._update_distance_ema(de2, dp2, ds2)
        de2n = de2 / (self.ema_de.to(de2.dtype) + self.eps)
        dp2n = dp2 / (self.ema_dp.to(dp2.dtype) + self.eps)
        ds2n = ds2 / (self.ema_ds.to(ds2.dtype) + self.eps)
        return de2n, dp2n, ds2n

    def _pack_output(
        self,
        x: torch.Tensor,
        logits: torch.Tensor,
        ze: torch.Tensor,
        yp: torch.Tensor,
        ys: torch.Tensor,
        de2: torch.Tensor,
        dp2: torch.Tensor,
        ds2: torch.Tensor,
        de2n: torch.Tensor,
        dp2n: torch.Tensor,
        ds2n: torch.Tensor,
        pe: torch.Tensor,
        pp: torch.Tensor,
        ps: torch.Tensor,
        branch_weights: torch.Tensor,
        temp: float,
        ze_raw: torch.Tensor,
    ) -> dict:
        return {
            "logits": logits,
            "ze": ze,
            "zp": yp,
            "zs": ys,
            "de2": de2,
            "dp2": dp2,
            "ds2": ds2,
            "de2n": de2n,
            "dp2n": dp2n,
            "ds2n": ds2n,
            "proto_e": pe,
            "proto_p": pp,
            "proto_s": ps,
            "route_signature": self._state_signature(x),
            "state": x,
            "state_norm": x.norm(dim=-1),
            "ze_norm": ze.norm(dim=-1),
            "ze_raw_norm": ze_raw.norm(dim=-1),
            "zp_norm": yp.norm(dim=-1),
            "zs_norm": ys.norm(dim=-1),
            "branch_weights": branch_weights,
            "radius_p": torch.clamp(self.r_p, min=0.1).detach(),
            "radius_s": torch.clamp(self.r_s, min=0.1).detach(),
            "temp": torch.tensor(temp, device=x.device, dtype=x.dtype),
            "ema_de": self.ema_de.detach().clone(),
            "ema_dp": self.ema_dp.detach().clone(),
            "ema_ds": self.ema_ds.detach().clone(),
        }

    def forward(self, x: torch.Tensor, return_dict: bool = False):
        if x.dim() != 2 or x.size(-1) != self.in_dim:
            raise ValueError(f"Expected x shape [B,{self.in_dim}], got {tuple(x.shape)}")

        if self.training:
            self.forward_count.add_(1)

        r_p = torch.clamp(self.r_p, min=0.1)
        r_s = torch.clamp(self.r_s, min=0.1)

        ze_raw = self.proj_e(x)
        ze = F.normalize(ze_raw, p=2, dim=-1, eps=self.eps)

        zp_tan = self.proj_p(x)
        zs_tan = self.proj_s(x)

        yp = self.M_p.exp_map_mu0(zp_tan)
        ys = self.M_s.exp_map_mu0(zs_tan)

        pe = F.normalize(self.proto_e_tan, p=2, dim=-1, eps=self.eps)
        pp = self.M_p.exp_map_mu0(self.proto_p_tan)
        ps = self.M_s.exp_map_mu0(self.proto_s_tan)

        de2 = torch.cdist(ze, pe).pow(2)

        c = P._c(r_p)
        dp = P.poincare_distance_c(yp[:, None, :], pp[None, :, :], c=c, keepdim=False)
        dp2 = dp.pow(2)

        K = SP._c(r_s)
        ds = SP.spherical_projected_gyro_distance(ys[:, None, :], ps[None, :, :], K=K).squeeze(-1)
        ds2 = ds.pow(2)

        de2n, dp2n, ds2n = self._normalize_distances(de2, dp2, ds2)
        bw = self._branch_weights()
        denom = max(self.temp, self.eps)
        logits = -(bw[0] * de2n + bw[1] * dp2n + bw[2] * ds2n) / denom

        if not return_dict:
            return logits

        return self._pack_output(
            x=x,
            logits=logits,
            ze=ze,
            yp=yp,
            ys=ys,
            de2=de2,
            dp2=dp2,
            ds2=ds2,
            de2n=de2n,
            dp2n=dp2n,
            ds2n=ds2n,
            pe=pe,
            pp=pp,
            ps=ps,
            branch_weights=bw,
            temp=denom,
            ze_raw=ze_raw,
        )
