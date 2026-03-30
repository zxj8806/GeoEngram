from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from MixCurv.ops import poincare as P
from MixCurv.ops import spherical_projected as SP

from .supervised import SequentialMNIST, eval_epoch, unpack_model_output
from .memory import EpisodicMemoryBuffer
from .generator import TopologyConditionedLatentGenerator, train_topology_generator


@dataclass
class CLResults:
    acc_matrix: List[List[Optional[float]]]
    forgetting: List[float]
    final_avg_acc: float
    avg_forgetting: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CLConfig:
    use_replay: bool = True
    replay_per_class: int = 20
    replay_batch_size: int = 64
    replay_weight: float = 1.0
    use_distill: bool = True
    geo_weight: float = 0.5
    route_weight: float = 0.5
    temp_weight: float = 0.25
    memory_warmup_batches: int = 32
    use_gen_replay: bool = False
    gen_replay_batch_size: int = 64
    gen_replay_weight: float = 1.0
    gen_route_weight: float = 0.5
    gen_noise_dim: int = 32
    gen_cond_dim: int = 64
    gen_hidden_dim: int = 256
    gen_train_epochs: int = 20
    gen_train_batch_size: int = 128
    gen_lr: float = 1e-3
    route_overlap_topk: int = 64
    distill_ramp_ratio: float = 0.2
    route_sep_weight: float = 0.3
    route_sep_margin: float = 0.5
    gate_sparse_weight: float = 1e-3
    gate_overlap_weight: float = 5e-2
    gate_binary_weight: float = 1e-3
    gate_smooth_weight: float = 1e-2


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _digits_to_indices(mnist_ds, digits: Sequence[int]) -> List[int]:
    digits_set = set(int(d) for d in digits)
    idxs = []
    targets = mnist_ds.targets
    for i in range(len(mnist_ds)):
        if int(targets[i]) in digits_set:
            idxs.append(i)
    return idxs


def make_split_seq_loaders(
    digits: Sequence[int],
    batch_size: int = 256,
    in_size: int = 16,
    num_workers: int = 2,
    data_root: str = "./data",
    limit_train: Optional[int] = None,
    limit_test: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    tfm = transforms.ToTensor()
    train_raw = datasets.MNIST(root=data_root, train=True, download=True, transform=tfm)
    test_raw = datasets.MNIST(root=data_root, train=False, download=True, transform=tfm)
    train_idxs = _digits_to_indices(train_raw, digits)
    test_idxs = _digits_to_indices(test_raw, digits)
    if limit_train is not None:
        train_idxs = train_idxs[: int(limit_train)]
    if limit_test is not None:
        test_idxs = test_idxs[: int(limit_test)]
    train_ds = SequentialMNIST(Subset(train_raw, train_idxs), in_size=in_size, perm=None)
    test_ds = SequentialMNIST(Subset(test_raw, test_idxs), in_size=in_size, perm=None)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    return train_loader, test_loader


def make_psmnist_seq_loaders(
    task_seed: int,
    batch_size: int = 256,
    in_size: int = 16,
    num_workers: int = 2,
    data_root: str = "./data",
    limit_train: Optional[int] = None,
    limit_test: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    tfm = transforms.ToTensor()
    train_raw = datasets.MNIST(root=data_root, train=True, download=True, transform=tfm)
    test_raw = datasets.MNIST(root=data_root, train=False, download=True, transform=tfm)
    g = torch.Generator().manual_seed(int(task_seed))
    perm = torch.randperm(784, generator=g)
    train_ds = SequentialMNIST(train_raw, in_size=in_size, perm=perm)
    test_ds = SequentialMNIST(test_raw, in_size=in_size, perm=perm)
    if limit_train is not None:
        train_ds = Subset(train_ds, list(range(int(limit_train))))
    if limit_test is not None:
        test_ds = Subset(test_ds, list(range(int(limit_test))))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    return train_loader, test_loader


def _format_matrix(acc_matrix: List[List[Optional[float]]]) -> str:
    lines = []
    n_tasks = len(acc_matrix)
    header = "      " + " ".join([f"T{k:02d}" for k in range(n_tasks)])
    lines.append(header)
    for t in range(n_tasks):
        row = []
        for k in range(n_tasks):
            v = acc_matrix[t][k]
            row.append("  --  " if v is None else f"{v*100:5.1f}")
        lines.append(f"Tr{t:02d}  " + " ".join(row))
    return "\n".join(lines)


def _forward_model(model: torch.nn.Module, x: torch.Tensor, task_id: Optional[torch.Tensor], return_aux: bool, **forward_kwargs):
    if task_id is not None:
        try:
            return model(x, task_id=task_id, return_aux=return_aux, **forward_kwargs)
        except TypeError:
            pass
    try:
        return model(x, return_aux=return_aux, **forward_kwargs)
    except TypeError:
        return model(x, **forward_kwargs)


@torch.no_grad()
def _eval_all_seen_tasks(model: torch.nn.Module, test_loaders: List[DataLoader], device: torch.device, forward_kwargs: Dict, task_ids: Sequence[int]) -> List[float]:
    accs = []
    for loader, task_id in zip(test_loaders, task_ids):
        try:
            _, acc = eval_epoch(model, loader, device, task_id=torch.full((1,), int(task_id), dtype=torch.long, device=device), **forward_kwargs)
        except TypeError:
            _, acc = eval_epoch(model, loader, device, **forward_kwargs)
        accs.append(float(acc))
    return accs


def _safe_radius(aux: Dict[str, Any], key: str, device: torch.device) -> torch.Tensor:
    val = aux.get(key)
    if isinstance(val, torch.Tensor):
        return val.detach().to(device=device).reshape(())
    return torch.tensor(1.0, device=device)


def _normalize_pairwise_distance(dmat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if dmat.dim() != 2:
        raise ValueError(f"got {tuple(dmat.shape)}")
    if dmat.size(0) <= 1:
        return dmat
    mask = ~torch.eye(dmat.size(0), dtype=torch.bool, device=dmat.device)
    vals = dmat.masked_select(mask)
    scale = vals.mean() if vals.numel() > 0 else dmat.mean()
    return dmat / (scale + eps)


def _pairwise_euclidean_distance(x: torch.Tensor) -> torch.Tensor:
    return _normalize_pairwise_distance(torch.cdist(x, x).pow(2))


def _pairwise_poincare_distance(x: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    c = P._c(radius)
    d = P.poincare_distance_c(x[:, None, :], x[None, :, :], c=c, keepdim=False)
    if d.dim() == 3 and d.size(-1) == 1:
        d = d.squeeze(-1)
    return _normalize_pairwise_distance(d.pow(2))


def _pairwise_spherical_distance(x: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    K = SP._c(radius)
    d = SP.spherical_projected_gyro_distance(x[:, None, :], x[None, :, :], K=K)
    if d.dim() == 3 and d.size(-1) == 1:
        d = d.squeeze(-1)
    return _normalize_pairwise_distance(d.pow(2))


def _branch_geodesic_loss(student: torch.Tensor, teacher: torch.Tensor, branch: str, radius: Optional[torch.Tensor] = None) -> torch.Tensor:
    if branch == "e":
        ds = _pairwise_euclidean_distance(student)
        dt = _pairwise_euclidean_distance(teacher)
    elif branch == "p":
        ds = _pairwise_poincare_distance(student, radius=radius)
        dt = _pairwise_poincare_distance(teacher, radius=radius)
    elif branch == "s":
        ds = _pairwise_spherical_distance(student, radius=radius)
        dt = _pairwise_spherical_distance(teacher, radius=radius)
    else:
        raise ValueError(f"Unknown branch: {branch}")
    return F.mse_loss(ds, dt)


def _route_support_loss(student_route: torch.Tensor, teacher_route: torch.Tensor, topk: int = 64, eps: float = 1e-6) -> torch.Tensor:
    k = max(1, min(int(topk), student_route.size(-1), teacher_route.size(-1)))
    s = F.normalize(torch.clamp(student_route, min=0.0), p=2, dim=-1, eps=eps)
    t = F.normalize(torch.clamp(teacher_route, min=0.0), p=2, dim=-1, eps=eps)
    teacher_vals, teacher_idx = torch.topk(t, k=k, dim=-1)
    student_support = torch.gather(s, dim=-1, index=teacher_idx)
    weights = teacher_vals / (teacher_vals.sum(dim=-1, keepdim=True) + eps)
    preserved = (student_support * weights).sum(dim=-1)
    return 1.0 - preserved.mean()


def _distill_loss(student_aux: Dict[str, Any], mem_batch: Dict[str, Any], geo_weight: float, route_weight: float, temp_weight: float, route_topk: int) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = next(v for v in student_aux.values() if isinstance(v, torch.Tensor)).device
    zero = torch.tensor(0.0, device=device)
    loss_geo_e = _branch_geodesic_loss(student_aux["ze"], mem_batch["ze"], branch="e") if "ze" in student_aux and "ze" in mem_batch else zero
    loss_geo_p = _branch_geodesic_loss(student_aux["zp"], mem_batch["zp"], branch="p", radius=_safe_radius(student_aux, "radius_p", device=device)) if "zp" in student_aux and "zp" in mem_batch else zero
    loss_geo_s = _branch_geodesic_loss(student_aux["zs"], mem_batch["zs"], branch="s", radius=_safe_radius(student_aux, "radius_s", device=device)) if "zs" in student_aux and "zs" in mem_batch else zero
    loss_geo = torch.stack([loss_geo_e, loss_geo_p, loss_geo_s]).mean()

    loss_route_cos = zero
    loss_route_sup = zero
    if "route_signature" in student_aux and "route_signature" in mem_batch:
        s_route = F.normalize(student_aux["route_signature"], p=2, dim=-1)
        t_route = F.normalize(mem_batch["route_signature"], p=2, dim=-1)
        loss_route_cos = 1.0 - F.cosine_similarity(s_route, t_route, dim=-1).mean()
        loss_route_sup = _route_support_loss(s_route, t_route, topk=route_topk)
    loss_route = 0.5 * loss_route_cos + 0.5 * loss_route_sup

    loss_temp = zero
    if "temporal_signature" in student_aux and "temporal_signature" in mem_batch:
        s_temp = F.normalize(student_aux["temporal_signature"], p=2, dim=-1)
        t_temp = F.normalize(mem_batch["temporal_signature"], p=2, dim=-1)
        loss_temp = 1.0 - F.cosine_similarity(s_temp, t_temp, dim=-1).mean()

    loss = geo_weight * loss_geo + route_weight * loss_route + temp_weight * loss_temp
    logs = {
        "loss_geo": float(loss_geo.detach().cpu()),
        "loss_geo_e": float(loss_geo_e.detach().cpu()),
        "loss_geo_p": float(loss_geo_p.detach().cpu()),
        "loss_geo_s": float(loss_geo_s.detach().cpu()),
        "loss_route": float(loss_route.detach().cpu()),
        "loss_route_cos": float(loss_route_cos.detach().cpu()),
        "loss_route_sup": float(loss_route_sup.detach().cpu()),
        "loss_temp": float(loss_temp.detach().cpu()),
    }
    return loss, logs


def _distill_ramp(epoch_idx: int, total_epochs: int, ratio: float) -> float:
    total_epochs = max(int(total_epochs), 1)
    warm = max(1, int(round(float(ratio) * total_epochs)))
    if epoch_idx <= warm:
        return float(max(epoch_idx - 1, 0) / max(warm, 1))
    return 1.0


def _task_class_route_centroids(memory: EpisodicMemoryBuffer, device: torch.device) -> Dict[Tuple[int, int], torch.Tensor]:
    return memory.task_class_route_centroids(device=device)


def _route_separation_loss(
    current_route: Optional[torch.Tensor],
    previous_route_centroids: List[torch.Tensor],
    *,
    margin: float = 0.5,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if current_route is None or len(previous_route_centroids) == 0:
        zero = torch.tensor(0.0, device=current_route.device if current_route is not None else torch.device("cpu"))
        return zero, {"route_sep_mean_sim": 0.0, "route_sep_max_sim": 0.0}
    curr = F.normalize(current_route, p=2, dim=-1, eps=eps).mean(dim=0)
    curr = F.normalize(curr, p=2, dim=-1, eps=eps)
    prev = torch.stack([F.normalize(c.to(device=curr.device), p=2, dim=-1, eps=eps) for c in previous_route_centroids], dim=0)
    sims = prev @ curr
    violations = F.relu(sims - float(margin))
    loss = (violations ** 2).mean()
    return loss, {
        "route_sep_mean_sim": float(sims.detach().mean().cpu()),
        "route_sep_max_sim": float(sims.detach().max().cpu()),
    }


def _make_gen_batch(memory: EpisodicMemoryBuffer, batch_size: int, device: torch.device) -> Optional[Dict[str, torch.Tensor]]:
    task_class_centroids = memory.task_class_route_centroids(device=device)
    if len(task_class_centroids) == 0:
        return None
    keys = sorted(task_class_centroids.keys())
    idx = torch.randint(low=0, high=len(keys), size=(int(batch_size),), device=device)
    task_ids = torch.tensor([keys[i][0] for i in idx.tolist()], dtype=torch.long, device=device)
    ys = torch.tensor([keys[i][1] for i in idx.tolist()], dtype=torch.long, device=device)
    routes = torch.stack([task_class_centroids[keys[i]] for i in idx.tolist()], dim=0)
    return {"task_id": task_ids, "y": ys, "route_signature": routes}


def _maybe_build_generator(memory: EpisodicMemoryBuffer, cl_cfg: CLConfig, device: torch.device, n_classes: int = 10, n_tasks: int = 5) -> Optional[TopologyConditionedLatentGenerator]:
    states = memory.stacked("state")
    routes = memory.stacked("route_signature")
    if states is None or routes is None:
        return None
    return TopologyConditionedLatentGenerator(
        state_dim=states.size(-1),
        route_dim=routes.size(-1),
        n_classes=n_classes,
        n_tasks=n_tasks,
        noise_dim=cl_cfg.gen_noise_dim,
        cond_dim=cl_cfg.gen_cond_dim,
        hidden_dim=cl_cfg.gen_hidden_dim,
    ).to(device)


@torch.no_grad()
def _collect_task_diagnostics(model: torch.nn.Module, loader: DataLoader, device: torch.device, forward_kwargs: Dict[str, Any], task_id: int, max_batches: int = 8) -> Dict[str, Any]:
    model.eval()
    sums: Dict[str, float] = defaultdict(float)
    count = 0
    for bidx, (x, _) in enumerate(loader):
        if bidx >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        task_tensor = torch.full((x.size(0),), int(task_id), dtype=torch.long, device=device)
        out = _forward_model(model, x, task_id=task_tensor, return_aux=True, **forward_kwargs)
        _, aux = unpack_model_output(out)
        for key in ["spike_mean", "dead_ratio", "state_norm", "rate_state_norm", "temporal_norm", "ze_norm", "ze_raw_norm", "zp_norm", "zs_norm", "gate_mean", "gate_active_frac"]:
            if key in aux and isinstance(aux[key], torch.Tensor):
                sums[key] += float(aux[key].detach().mean().cpu())
        bw = aux.get("branch_weights")
        if isinstance(bw, torch.Tensor) and bw.numel() == 3:
            sums["branch_w_e"] += float(bw[0].detach().cpu())
            sums["branch_w_p"] += float(bw[1].detach().cpu())
            sums["branch_w_s"] += float(bw[2].detach().cpu())
        count += 1
    return {k: v / max(count, 1) for k, v in sums.items()}


def _memory_state_for_replay(mem_batch: Dict[str, Any]) -> Optional[torch.Tensor]:
    state = mem_batch.get("engram_state")
    if state is None:
        state = mem_batch.get("state")
    return state


def _forward_replay_from_memory_state(model: torch.nn.Module, mem_batch: Dict[str, Any], use_aux: bool):
    if not hasattr(model, "forward_from_state"):
        return None
    state = _memory_state_for_replay(mem_batch)
    if state is None:
        return None
    kwargs: Dict[str, Any] = {"return_aux": use_aux}
    for key in ["route_signature", "temporal_signature", "rate_state"]:
        if key in mem_batch:
            kwargs[key] = mem_batch[key]
    return model.forward_from_state(state, **kwargs)


def _sample_memory_targets(memory: EpisodicMemoryBuffer, batch_size: int, device: torch.device) -> Tuple[Optional[Dict[str, Any]], str]:
    bank_batch = memory.sample_engram_bank(batch_size=batch_size, device=device)
    if bank_batch is not None and ("engram_state" in bank_batch or "state" in bank_batch):
        return bank_batch, "engram_bank"
    raw_batch = memory.sample(batch_size=batch_size, device=device)
    return raw_batch, "sample_buffer"


def _train_task_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    forward_kwargs: Dict[str, Any],
    cl_cfg: CLConfig,
    memory: EpisodicMemoryBuffer,
    use_aux: bool,
    generator: Optional[TopologyConditionedLatentGenerator] = None,
    *,
    current_task_id: int,
    epoch_idx: int = 1,
    total_epochs: int = 1,
) -> Dict[str, float]:
    model.train()
    sums: Dict[str, float] = defaultdict(float)
    n_samples = 0
    distill_scale = _distill_ramp(epoch_idx=epoch_idx, total_epochs=total_epochs, ratio=cl_cfg.distill_ramp_ratio)

    prev_route_centroids_map = _task_class_route_centroids(memory, device=device) if cl_cfg.route_sep_weight > 0 and not memory.is_empty else {}
    prev_route_centroids = [
        centroid for (task_id, _class_id), centroid in prev_route_centroids_map.items()
        if int(task_id) != int(current_task_id)
    ]

    for x, y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        task_tensor = torch.full((x.size(0),), int(current_task_id), dtype=torch.long, device=device)
        out = _forward_model(model, x, task_id=task_tensor, return_aux=use_aux, **forward_kwargs)
        logits, aux = unpack_model_output(out)
        loss_task = F.cross_entropy(logits, y)
        loss_total = loss_task

        replay_loss = torch.tensor(0.0, device=device)
        route_sep_loss = torch.tensor(0.0, device=device)
        gate_loss = torch.tensor(0.0, device=device)
        gen_replay_loss = torch.tensor(0.0, device=device)
        gen_route_loss = torch.tensor(0.0, device=device)
        bank_source_flag = 0.0
        distill_logs = {
            "loss_geo": 0.0,
            "loss_geo_e": 0.0,
            "loss_geo_p": 0.0,
            "loss_geo_s": 0.0,
            "loss_route": 0.0,
            "loss_route_cos": 0.0,
            "loss_route_sup": 0.0,
            "loss_temp": 0.0,
        }
        route_sep_logs = {"route_sep_mean_sim": 0.0, "route_sep_max_sim": 0.0}

        if (cl_cfg.use_replay or cl_cfg.use_distill) and not memory.is_empty:
            mem_batch, mem_source = _sample_memory_targets(memory, batch_size=cl_cfg.replay_batch_size, device=device)
            if mem_batch is not None:
                bank_source_flag = 1.0 if mem_source == "engram_bank" else 0.0

                if cl_cfg.use_replay:
                    replay_out = _forward_replay_from_memory_state(model, mem_batch, use_aux=use_aux)
                    if replay_out is None and "x" in mem_batch:
                        replay_out = _forward_model(model, mem_batch["x"], task_id=mem_batch.get("task_id"), return_aux=use_aux, **forward_kwargs)
                    if replay_out is not None:
                        replay_logits, _ = unpack_model_output(replay_out)
                        replay_loss = F.cross_entropy(replay_logits, mem_batch["y"])
                        loss_total = loss_total + cl_cfg.replay_weight * replay_loss

                if cl_cfg.use_distill:
                    mem_out = _forward_replay_from_memory_state(model, mem_batch, use_aux=use_aux)
                    if mem_out is None and "x" in mem_batch:
                        mem_out = _forward_model(model, mem_batch["x"], task_id=mem_batch.get("task_id"), return_aux=use_aux, **forward_kwargs)
                    if mem_out is not None:
                        _, mem_aux = unpack_model_output(mem_out)
                        if mem_aux:
                            distill_term, distill_logs = _distill_loss(
                                mem_aux,
                                mem_batch,
                                geo_weight=cl_cfg.geo_weight,
                                route_weight=cl_cfg.route_weight,
                                temp_weight=cl_cfg.temp_weight,
                                route_topk=cl_cfg.route_overlap_topk,
                            )
                            loss_total = loss_total + distill_scale * distill_term

        if cl_cfg.route_sep_weight > 0 and len(prev_route_centroids) > 0 and aux and "route_signature" in aux:
            route_sep_loss, route_sep_logs = _route_separation_loss(
                current_route=aux["route_signature"],
                previous_route_centroids=prev_route_centroids,
                margin=cl_cfg.route_sep_margin,
            )
            loss_total = loss_total + distill_scale * cl_cfg.route_sep_weight * route_sep_loss

        reservoir = getattr(model, "reservoir", None)
        gate_logs = {
            "loss_gate_sparse": 0.0,
            "loss_gate_overlap": 0.0,
            "loss_gate_binary": 0.0,
            "loss_gate_smooth": 0.0,
            "gate_mean": 0.0,
            "gate_min": 0.0,
            "gate_max": 0.0,
            "gate_active_frac": 0.0,
            "gate_overlap_mean": 0.0,
            "gate_overlap_max": 0.0,
        }
        if reservoir is not None and hasattr(reservoir, "gate_regularization"):
            gate_loss, gate_logs = reservoir.gate_regularization(
                task_id=int(current_task_id),
                previous_task_ids=list(range(int(current_task_id))),
                sparsity_weight=cl_cfg.gate_sparse_weight,
                overlap_weight=cl_cfg.gate_overlap_weight,
                binary_weight=cl_cfg.gate_binary_weight,
                smooth_weight=cl_cfg.gate_smooth_weight,
            )
            loss_total = loss_total + gate_loss

        if cl_cfg.use_gen_replay and generator is not None and hasattr(model, "forward_from_state") and not memory.is_empty:
            gen_cond = _make_gen_batch(memory, cl_cfg.gen_replay_batch_size, device)
            if gen_cond is not None:
                gen_samples = generator.sample(y=gen_cond["y"], task_id=gen_cond["task_id"], route_cond=gen_cond["route_signature"])
                gen_out = model.forward_from_state(
                    gen_samples["state"],
                    return_aux=True,
                    route_signature=gen_samples["route_signature"],
                    temporal_signature=gen_samples["temporal_signature"],
                )
                gen_logits, gen_aux = unpack_model_output(gen_out)
                gen_replay_loss = F.cross_entropy(gen_logits, gen_cond["y"])
                if "route_signature" in gen_aux:
                    gen_route_loss = F.mse_loss(gen_aux["route_signature"], gen_cond["route_signature"])
                loss_total = loss_total + cl_cfg.gen_replay_weight * gen_replay_loss + cl_cfg.gen_route_weight * gen_route_loss

        optimizer.zero_grad(set_to_none=True)
        loss_total.backward()
        optimizer.step()

        bs = x.size(0)
        n_samples += bs
        sums["loss_total"] += float(loss_total.detach().cpu()) * bs
        sums["loss_task"] += float(loss_task.detach().cpu()) * bs
        sums["loss_replay"] += float(replay_loss.detach().cpu()) * bs
        sums["loss_geo"] += distill_logs["loss_geo"] * bs
        sums["loss_geo_e"] += distill_logs["loss_geo_e"] * bs
        sums["loss_geo_p"] += distill_logs["loss_geo_p"] * bs
        sums["loss_geo_s"] += distill_logs["loss_geo_s"] * bs
        sums["loss_route"] += distill_logs["loss_route"] * bs
        sums["loss_route_cos"] += distill_logs["loss_route_cos"] * bs
        sums["loss_route_sup"] += distill_logs["loss_route_sup"] * bs
        sums["loss_temp"] += distill_logs["loss_temp"] * bs
        sums["loss_route_sep"] += float(route_sep_loss.detach().cpu()) * bs
        sums["route_sep_mean_sim"] += route_sep_logs["route_sep_mean_sim"] * bs
        sums["route_sep_max_sim"] += route_sep_logs["route_sep_max_sim"] * bs
        sums["loss_gate"] += float(gate_loss.detach().cpu()) * bs
        sums["loss_gate_sparse"] += gate_logs["loss_gate_sparse"] * bs
        sums["loss_gate_overlap"] += gate_logs["loss_gate_overlap"] * bs
        sums["loss_gate_binary"] += gate_logs["loss_gate_binary"] * bs
        sums["loss_gate_smooth"] += gate_logs["loss_gate_smooth"] * bs
        sums["gate_mean"] += gate_logs["gate_mean"] * bs
        sums["gate_min"] += gate_logs["gate_min"] * bs
        sums["gate_max"] += gate_logs["gate_max"] * bs
        sums["gate_active_frac"] += gate_logs["gate_active_frac"] * bs
        sums["gate_overlap_mean"] += gate_logs["gate_overlap_mean"] * bs
        sums["gate_overlap_max"] += gate_logs["gate_overlap_max"] * bs
        sums["loss_gen_replay"] += float(gen_replay_loss.detach().cpu()) * bs
        sums["loss_gen_route"] += float(gen_route_loss.detach().cpu()) * bs
        sums["distill_scale"] += float(distill_scale) * bs
        sums["use_engram_bank"] += float(bank_source_flag) * bs
        pred = torch.argmax(logits.detach(), dim=1)
        sums["acc"] += float((pred == y).float().mean().cpu()) * bs
        if aux:
            for key in ["spike_mean", "dead_ratio", "state_norm", "rate_state_norm", "temporal_norm", "ze_norm", "ze_raw_norm", "zp_norm", "zs_norm", "gate_mean", "gate_active_frac"]:
                if key in aux and isinstance(aux[key], torch.Tensor):
                    sums[key] += float(aux[key].detach().mean().cpu()) * bs
    return {k: v / max(n_samples, 1) for k, v in sums.items()}


@torch.no_grad()
def _populate_memory(model: torch.nn.Module, train_loader: DataLoader, memory: EpisodicMemoryBuffer, device: torch.device, forward_kwargs: Dict[str, Any], max_batches: int, task_id: int) -> None:
    model.eval()
    for bidx, (x, y) in enumerate(train_loader):
        if bidx >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        task_tensor = torch.full((x.size(0),), int(task_id), dtype=torch.long, device=device)
        out = _forward_model(model, x, task_id=task_tensor, return_aux=True, **forward_kwargs)
        logits, aux = unpack_model_output(out)
        aux_to_store = dict(aux)
        aux_to_store["teacher_logits"] = logits.detach()
        memory.add_batch(x, y, aux_to_store, task_id=task_id)


def run_continual_learning(
    model: torch.nn.Module,
    device: torch.device,
    make_optimizer_fn,
    forward_kwargs: Dict,
    *,
    n_tasks: int = 5,
    benchmark: str = "psmnist",
    base_seed: int = 1111,
    digits_per_task: int = 2,
    epochs_per_task: int = 5,
    batch_size: int = 256,
    in_size: int = 16,
    num_workers: int = 2,
    data_root: str = "./data",
    limit_train: Optional[int] = None,
    limit_test: Optional[int] = None,
    reset_optimizer_each_task: bool = False,
    out_dir: str = "./cl_logs",
    verbose: bool = True,
    use_replay: bool = True,
    replay_per_class: int = 20,
    replay_batch_size: int = 64,
    replay_weight: float = 1.0,
    use_distill: bool = True,
    geo_weight: float = 0.5,
    route_weight: float = 0.5,
    temp_weight: float = 0.25,
    memory_warmup_batches: int = 32,
    use_gen_replay: bool = False,
    gen_replay_batch_size: int = 64,
    gen_replay_weight: float = 1.0,
    gen_route_weight: float = 0.5,
    gen_noise_dim: int = 32,
    gen_cond_dim: int = 64,
    gen_hidden_dim: int = 256,
    gen_train_epochs: int = 20,
    gen_train_batch_size: int = 128,
    gen_lr: float = 1e-3,
    route_overlap_topk: int = 64,
    route_sep_weight: float = 0.3,
    route_sep_margin: float = 0.5,
    gate_sparse_weight: float = 1e-3,
    gate_overlap_weight: float = 5e-2,
    gate_binary_weight: float = 1e-3,
    gate_smooth_weight: float = 1e-2,
) -> CLResults:
    _ensure_dir(out_dir)
    benchmark = str(benchmark).lower().strip()
    train_loaders: List[DataLoader] = []
    test_loaders: List[DataLoader] = []
    task_meta: List[str] = []
    for t in range(n_tasks):
        if benchmark == "psmnist":
            tr, te = make_psmnist_seq_loaders(
                task_seed=base_seed + t,
                batch_size=batch_size,
                in_size=in_size,
                num_workers=num_workers,
                data_root=data_root,
                limit_train=limit_train,
                limit_test=limit_test,
            )
            task_meta.append(f"psmnist(seed={base_seed+t})")
        else:
            start = t * digits_per_task
            digits = list(range(start, start + digits_per_task))
            tr, te = make_split_seq_loaders(
                digits=digits,
                batch_size=batch_size,
                in_size=in_size,
                num_workers=num_workers,
                data_root=data_root,
                limit_train=limit_train,
                limit_test=limit_test,
            )
            task_meta.append(f"splitmnist(digits={digits})")
        train_loaders.append(tr)
        test_loaders.append(te)

    cl_cfg = CLConfig(
        use_replay=bool(use_replay),
        replay_per_class=int(replay_per_class),
        replay_batch_size=int(replay_batch_size),
        replay_weight=float(replay_weight),
        use_distill=bool(use_distill),
        geo_weight=float(geo_weight),
        route_weight=float(route_weight),
        temp_weight=float(temp_weight),
        memory_warmup_batches=int(memory_warmup_batches),
        use_gen_replay=bool(use_gen_replay),
        gen_replay_batch_size=int(gen_replay_batch_size),
        gen_replay_weight=float(gen_replay_weight),
        gen_route_weight=float(gen_route_weight),
        gen_noise_dim=int(gen_noise_dim),
        gen_cond_dim=int(gen_cond_dim),
        gen_hidden_dim=int(gen_hidden_dim),
        gen_train_epochs=int(gen_train_epochs),
        gen_train_batch_size=int(gen_train_batch_size),
        gen_lr=float(gen_lr),
        route_overlap_topk=int(route_overlap_topk),
        route_sep_weight=float(route_sep_weight),
        route_sep_margin=float(route_sep_margin),
        gate_sparse_weight=float(gate_sparse_weight),
        gate_overlap_weight=float(gate_overlap_weight),
        gate_binary_weight=float(gate_binary_weight),
        gate_smooth_weight=float(gate_smooth_weight),
    )
    memory = EpisodicMemoryBuffer(per_class=cl_cfg.replay_per_class, device="cpu")
    generator: Optional[TopologyConditionedLatentGenerator] = None
    optim = make_optimizer_fn()

    for t in range(n_tasks):
        if reset_optimizer_each_task and t > 0:
            optim = make_optimizer_fn()
        if verbose:
            print("\n" + "=" * 80)
            print(f"[Task {t+1}/{n_tasks}] {task_meta[t]}")
            print("=" * 80)
        epoch_logs: List[Dict[str, float]] = []
        for ep in range(1, epochs_per_task + 1):
            tr_stats = _train_task_epoch(
                model=model,
                train_loader=train_loaders[t],
                optimizer=optim,
                device=device,
                forward_kwargs=forward_kwargs,
                cl_cfg=cl_cfg,
                memory=memory,
                use_aux=True,
                generator=generator,
                current_task_id=t,
                epoch_idx=ep,
                total_epochs=epochs_per_task,
            )
            epoch_logs.append(tr_stats)
            if verbose:
                print(
                    f"  epoch {ep:02d}/{epochs_per_task}: "
                    f"loss={tr_stats.get('loss_total', 0.0):.4f} "
                )

        reservoir = getattr(model, "reservoir", None)
        if reservoir is not None and hasattr(reservoir, "freeze_task_gate"):
            reservoir.freeze_task_gate(task_id=t)

        _populate_memory(
            model=model,
            train_loader=train_loaders[t],
            memory=memory,
            device=device,
            forward_kwargs=forward_kwargs,
            max_batches=cl_cfg.memory_warmup_batches,
            task_id=t,
        )

        if cl_cfg.use_gen_replay and len(memory) > 0 and hasattr(model, "forward_from_state"):
            if generator is None:
                generator = _maybe_build_generator(memory, cl_cfg, device, n_classes=10, n_tasks=n_tasks)
            if generator is not None:
                train_topology_generator(
                    generator=generator,
                    memory=memory,
                    device=device,
                    epochs=cl_cfg.gen_train_epochs,
                    batch_size=cl_cfg.gen_train_batch_size,
                    lr=cl_cfg.gen_lr,
                )

        task_diag = _collect_task_diagnostics(model, test_loaders[t], device, forward_kwargs, task_id=t, max_batches=8)

        accs = _eval_all_seen_tasks(model, test_loaders[: t + 1], device, forward_kwargs, task_ids=list(range(t + 1)))


