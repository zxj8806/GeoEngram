from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GeneratorTrainStats:
    loss_total: float
    loss_state: float
    loss_route: float
    loss_consistency: float


class TopologyConditionedLatentGenerator(nn.Module):
    def __init__(self, state_dim: int, n_classes: int, n_tasks: int, route_dim: Optional[int] = None, noise_dim: int = 32, cond_dim: int = 64, hidden_dim: int = 256) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.route_dim = int(route_dim if route_dim is not None else state_dim)
        self.n_classes = int(n_classes)
        self.n_tasks = int(n_tasks)
        self.noise_dim = int(noise_dim)
        self.cond_dim = int(cond_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_bins = max(1, self.state_dim // max(self.route_dim, 1)) if self.state_dim % max(self.route_dim, 1) == 0 else 1
        self.class_emb = nn.Embedding(self.n_classes, self.cond_dim)
        self.task_emb = nn.Embedding(self.n_tasks, self.cond_dim)
        self.route_encoder = nn.Sequential(nn.Linear(self.route_dim, self.cond_dim), nn.ReLU(inplace=True), nn.Linear(self.cond_dim, self.cond_dim))
        self.net = nn.Sequential(nn.Linear(self.noise_dim + 3 * self.cond_dim, self.hidden_dim), nn.ReLU(inplace=True), nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(inplace=True))
        self.state_head = nn.Linear(self.hidden_dim, self.state_dim)
        self.route_head = nn.Linear(self.hidden_dim, self.route_dim)

    def forward(self, y: torch.Tensor, task_id: torch.Tensor, route_cond: torch.Tensor, z: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if z is None:
            z = torch.randn(y.size(0), self.noise_dim, device=y.device, dtype=route_cond.dtype)
        c = self.class_emb(y)
        t = self.task_emb(task_id)
        r = self.route_encoder(route_cond)
        h = self.net(torch.cat([z, c, t, r], dim=-1))
        state = self.state_head(h)
        route = F.normalize(self.route_head(h), p=2, dim=-1)
        temporal_signature = F.normalize(state, p=2, dim=-1)
        return {"state": state, "route_signature": route, "temporal_signature": temporal_signature}

    @torch.no_grad()
    def sample(self, y: torch.Tensor, task_id: torch.Tensor, route_cond: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.eval()
        return self.forward(y=y, task_id=task_id, route_cond=route_cond, z=None)


def _state_to_route_proxy(state: torch.Tensor, route_dim: int, num_bins: int) -> torch.Tensor:
    if num_bins <= 1 or state.size(-1) != route_dim * num_bins:
        proxy = state[..., :route_dim]
    else:
        proxy = state.view(state.size(0), num_bins, route_dim).mean(dim=1)
    return F.normalize(proxy, p=2, dim=-1)


def train_topology_generator(generator: TopologyConditionedLatentGenerator, memory, device: torch.device, *, epochs: int = 20, batch_size: int = 128, lr: float = 1e-3) -> GeneratorTrainStats:
    if memory.is_empty:
        return GeneratorTrainStats(0.0, 0.0, 0.0, 0.0)
    labels = memory.labels_tensor(device=device)
    task_ids = memory.task_ids_tensor(device=device)
    states = memory.stacked("state", device=device)
    routes = memory.stacked("route_signature", device=device)
    if labels is None or task_ids is None or states is None or routes is None:
        return GeneratorTrainStats(0.0, 0.0, 0.0, 0.0)
    generator.to(device)
    generator.train()
    opt = torch.optim.Adam(generator.parameters(), lr=lr)
    n = labels.size(0)
    if n == 0:
        return GeneratorTrainStats(0.0, 0.0, 0.0, 0.0)
    sum_total = sum_state = sum_route = sum_cons = 0.0
    steps = 0
    for _ in range(max(int(epochs), 1)):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, max(int(batch_size), 1)):
            idx = perm[start:start + max(int(batch_size), 1)]
            yb, tb, sb, rb = labels[idx], task_ids[idx], states[idx], routes[idx]
            out = generator(y=yb, task_id=tb, route_cond=rb)
            loss_state = F.mse_loss(out["state"], sb)
            loss_route = F.mse_loss(out["route_signature"], rb)
            route_proxy = _state_to_route_proxy(out["state"], generator.route_dim, generator.num_bins)
            loss_cons = F.mse_loss(route_proxy, rb)
            loss = loss_state + 0.5 * loss_route + 0.25 * loss_cons
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bs = yb.size(0)
            sum_total += float(loss.detach().cpu()) * bs
            sum_state += float(loss_state.detach().cpu()) * bs
            sum_route += float(loss_route.detach().cpu()) * bs
            sum_cons += float(loss_cons.detach().cpu()) * bs
            steps += bs
    denom = max(steps, 1)
    return GeneratorTrainStats(loss_total=sum_total / denom, loss_state=sum_state / denom, loss_route=sum_route / denom, loss_consistency=sum_cons / denom)
