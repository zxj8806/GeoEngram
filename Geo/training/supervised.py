import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Optional, Dict, Any, Tuple

from Geo.encoder import PoissonMNIST


class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, thresh: float, lens: float):
        ctx.save_for_backward(input)
        ctx.thresh = float(thresh)
        ctx.lens = float(lens)
        return input.gt(ctx.thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        thresh = ctx.thresh
        lens = ctx.lens
        grad_input = grad_output.clone()
        temp = (torch.abs(input - thresh) < lens).float()
        return grad_input * temp, None, None


act_fun = ActFun.apply


class SequentialMNIST(torch.utils.data.Dataset):
    def __init__(self, mnist_ds, in_size=16, perm=None):
        self.mnist = mnist_ds
        self.in_size = int(in_size)
        assert 784 % self.in_size == 0, ""
        self.T = 784 // self.in_size
        self.perm = perm

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        x = img.view(-1)
        if self.perm is not None:
            x = x[self.perm]
        x = x.view(self.T, self.in_size)
        return x, label


def make_mnist_seq_loaders(batch_size=256, in_size=16, permute=False, seed=1111, limit_train=None, limit_test=None, num_workers=2, data_root="./data"):
    tfm = transforms.ToTensor()
    train_raw = datasets.MNIST(root=data_root, train=True, download=True, transform=tfm)
    test_raw = datasets.MNIST(root=data_root, train=False, download=True, transform=tfm)
    perm = None
    if permute:
        g = torch.Generator().manual_seed(int(seed))
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


def make_mnist_poisson_loaders(batch_size=128, time_steps=20, p_scale=0.25, limit_train=None, limit_test=None, num_workers=2, data_root="./data"):
    tfm = transforms.ToTensor()
    train_raw = datasets.MNIST(root=data_root, train=True, download=True, transform=tfm)
    test_raw = datasets.MNIST(root=data_root, train=False, download=True, transform=tfm)
    if limit_train is not None:
        train_raw = Subset(train_raw, list(range(int(limit_train))))
    if limit_test is not None:
        test_raw = Subset(test_raw, list(range(int(limit_test))))
    enc = PoissonMNIST(time_steps=time_steps, p_scale=p_scale, seed=123)
    def collate(batch):
        imgs = torch.stack([b[0] for b in batch], dim=0)
        y = torch.tensor([b[1] for b in batch], dtype=torch.long)
        x = imgs.view(imgs.size(0), -1)
        spikes = enc.encode_image_batch(x)
        return spikes, y
    train_loader = DataLoader(train_raw, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False, collate_fn=collate)
    test_loader = DataLoader(test_raw, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False, collate_fn=collate)
    return train_loader, test_loader


def unpack_model_output(output) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if isinstance(output, dict):
        logits = output.get("logits")
        if logits is None:
            raise ValueError("")
        aux = output.get("aux", {})
        return logits, aux
    return output, {}


def _topk_normalized_signature(x: torch.Tensor, topk: int = 64, eps: float = 1e-6) -> torch.Tensor:
    k = max(1, min(int(topk), x.size(-1)))
    vals, idx = torch.topk(x, k=k, dim=-1)
    sig = torch.zeros_like(x)
    sig.scatter_(dim=-1, index=idx, src=vals)
    sig = F.normalize(sig, p=2, dim=-1, eps=eps)
    return sig


def structural_route_signature(spike_rec: torch.Tensor, w_latent: torch.Tensor, mask_latent: Optional[torch.Tensor] = None, *, task_gate: Optional[torch.Tensor] = None, topk: int = 64, eps: float = 1e-6) -> torch.Tensor:
    if spike_rec.dim() != 3:
        raise ValueError(f"got {tuple(spike_rec.shape)}")
    B, T, N = spike_rec.shape
    W = w_latent
    if mask_latent is not None:
        W = W * mask_latent
    W = W.abs().to(spike_rec.dtype)
    if task_gate is None:
        gate = torch.ones(B, N, device=spike_rec.device, dtype=spike_rec.dtype)
    else:
        gate = task_gate.to(device=spike_rec.device, dtype=spike_rec.dtype)
        if gate.dim() == 1:
            gate = gate.unsqueeze(0).expand(B, -1)
    gated_spike = spike_rec * gate[:, None, :]
    if T < 2:
        route_state = gated_spike.sum(dim=1)
        return _topk_normalized_signature(route_state, topk=topk, eps=eps)
    edge_gate = gate.unsqueeze(-1) * gate.unsqueeze(-2)
    Wb = W.unsqueeze(0) * edge_gate
    prev_spk = gated_spike[:, :-1, :]
    curr_spk = gated_spike[:, 1:, :]
    routed = torch.einsum("btn,bnm->btm", prev_spk, Wb) * curr_spk
    route_state = routed.sum(dim=1)
    state_fallback = gated_spike.mean(dim=1)
    zero_mask = route_state.abs().sum(dim=-1, keepdim=True) <= eps
    route_state = torch.where(zero_mask, state_fallback, route_state)
    return _topk_normalized_signature(route_state, topk=topk, eps=eps)


def temporal_binned_engram(spike_rec: torch.Tensor, num_bins: int = 4, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if spike_rec.dim() != 3:
        raise ValueError(f"got {tuple(spike_rec.shape)}")
    B, T, N = spike_rec.shape
    bins = max(1, int(num_bins))
    chunks = torch.chunk(spike_rec, chunks=bins, dim=1)
    pooled = []
    for ch in chunks:
        pooled.append(ch.mean(dim=1) if ch.size(1) > 0 else torch.zeros(B, N, device=spike_rec.device, dtype=spike_rec.dtype))
    engram_state = torch.cat(pooled, dim=-1)
    temporal_signature = F.normalize(engram_state, p=2, dim=-1, eps=eps)
    rate_state = spike_rec.mean(dim=1)
    return engram_state, temporal_signature, rate_state


class ReservoirRateClassifier(nn.Module):
    def __init__(self, reservoir, n_classes=10, readout: Optional[nn.Module] = None, engram_bins: int = 4):
        super().__init__()
        self.reservoir = reservoir
        self.n_classes = int(n_classes)
        self.engram_bins = int(engram_bins)
        self.readout = readout if readout is not None else nn.Linear(reservoir.n_neurons * self.engram_bins, n_classes)

    def _route_topk(self) -> int:
        return int(getattr(self.readout, "route_topk", 64))

    def forward_from_state(self, state: torch.Tensor, return_aux: bool = False, route_signature: Optional[torch.Tensor] = None, temporal_signature: Optional[torch.Tensor] = None, rate_state: Optional[torch.Tensor] = None):
        if return_aux:
            try:
                head_out = self.readout(state, return_dict=True)
                logits = head_out["logits"]
                aux = {k: v for k, v in head_out.items() if k != "logits"}
            except TypeError:
                logits = self.readout(state)
                aux = {}
            aux["state"] = state
            aux["engram_state"] = state
            aux["route_signature"] = route_signature if route_signature is not None else F.normalize(state, p=2, dim=-1)
            aux["temporal_signature"] = temporal_signature if temporal_signature is not None else F.normalize(state, p=2, dim=-1)
            aux["state_norm"] = state.norm(dim=-1)
            if rate_state is not None:
                aux["rate_state"] = rate_state
                aux["rate_state_norm"] = rate_state.norm(dim=-1)
            return {"logits": logits, "aux": aux}
        return self.readout(state)

    def forward(self, X, mem_thr=0.3, tau_mem=20.0, decay=None, alpha=2.0, lens=0.2, return_aux: bool = False, task_id: Optional[torch.Tensor] = None):
        spk = self.reservoir(X, mem_thr=mem_thr, tau_mem=tau_mem, decay=decay, alpha=alpha, lens=lens, task_id=task_id)
        engram_state, temporal_signature, rate_state = temporal_binned_engram(spk, num_bins=self.engram_bins)
        gate = self.reservoir.task_gate(task_id=task_id, batch=X.size(0)) if task_id is not None and hasattr(self.reservoir, "task_gate") else None
        route_signature = structural_route_signature(spk, self.reservoir.w_latent, getattr(self.reservoir, "mask_latent", None), task_gate=gate, topk=self._route_topk())
        if not return_aux:
            return self.forward_from_state(engram_state, return_aux=False)
        out = self.forward_from_state(engram_state, return_aux=True, route_signature=route_signature, temporal_signature=temporal_signature, rate_state=rate_state)
        aux = out["aux"]
        aux.update({"spike_mean": spk.mean(dim=(1, 2)), "dead_ratio": (rate_state <= 1e-6).float().mean(dim=-1), "temporal_norm": temporal_signature.norm(dim=-1)})
        if gate is not None:
            aux["gate_mean"] = gate.mean(dim=-1)
            aux["gate_min"] = gate.min(dim=-1).values
            aux["gate_max"] = gate.max(dim=-1).values
            aux["gate_active_frac"] = (gate > 0.5).float().mean(dim=-1)
        return {"logits": out["logits"], "aux": aux}


class ReservoirTemporalReadoutClassifier(nn.Module):
    def __init__(self, reservoir, n_classes=10, readout: Optional[nn.Module] = None, readout_mode: str = "temporal", engram_bins: int = 4):
        super().__init__()
        self.reservoir = reservoir
        self.n_classes = int(n_classes)
        self.engram_bins = int(engram_bins)
        self.readout = readout if readout is not None else nn.Linear(reservoir.n_neurons * self.engram_bins, n_classes)
        self.readout_mode = str(readout_mode).lower().strip()
        if self.readout_mode not in ("temporal", "rate"):
            raise ValueError("")

    def _route_topk(self) -> int:
        return int(getattr(self.readout, "route_topk", 64))

    def forward_from_state(self, state: torch.Tensor, return_aux: bool = False, route_signature: Optional[torch.Tensor] = None, temporal_signature: Optional[torch.Tensor] = None, rate_state: Optional[torch.Tensor] = None):
        if return_aux:
            try:
                head_out = self.readout(state, return_dict=True)
                logits = head_out["logits"]
                aux = {k: v for k, v in head_out.items() if k != "logits"}
            except TypeError:
                logits = self.readout(state)
                aux = {}
            aux["state"] = state
            aux["engram_state"] = state
            aux["route_signature"] = route_signature if route_signature is not None else F.normalize(state, p=2, dim=-1)
            aux["temporal_signature"] = temporal_signature if temporal_signature is not None else F.normalize(state, p=2, dim=-1)
            aux["state_norm"] = state.norm(dim=-1)
            if rate_state is not None:
                aux["rate_state"] = rate_state
                aux["rate_state_norm"] = rate_state.norm(dim=-1)
            return {"logits": logits, "aux": aux}
        return self.readout(state)

    def forward(self, X_seq, mem_thr=0.3, decay=0.9, alpha=2.0, lens=0.2, return_aux: bool = False, task_id: Optional[torch.Tensor] = None):
        B, T, _ = X_seq.shape
        mem, spk = self.reservoir.reset_state(B)
        gate = self.reservoir.task_gate(task_id=task_id, batch=B) if task_id is not None and hasattr(self.reservoir, "task_gate") else None
        spikes = []
        for t in range(T):
            mem, spk = self.reservoir.step(X_seq[:, t, :], mem, spk, mem_thr=mem_thr, decay=decay, alpha=alpha, lens=lens, task_id=task_id)
            spikes.append(spk)
        spike_rec = torch.stack(spikes, dim=1)
        engram_state, temporal_signature, rate_state = temporal_binned_engram(spike_rec, num_bins=self.engram_bins)
        route_signature = structural_route_signature(spike_rec, self.reservoir.w_latent, getattr(self.reservoir, "mask_latent", None), task_gate=gate, topk=self._route_topk())
        if not return_aux:
            return self.forward_from_state(engram_state, return_aux=False)
        out = self.forward_from_state(engram_state, return_aux=True, route_signature=route_signature, temporal_signature=temporal_signature, rate_state=rate_state)
        aux = out["aux"]
        aux.update({"spike_mean": spike_rec.mean(dim=(1, 2)), "dead_ratio": (rate_state <= 1e-6).float().mean(dim=-1), "temporal_norm": temporal_signature.norm(dim=-1)})
        if gate is not None:
            aux["gate_mean"] = gate.mean(dim=-1)
            aux["gate_min"] = gate.min(dim=-1).values
            aux["gate_max"] = gate.max(dim=-1).values
            aux["gate_active_frac"] = (gate > 0.5).float().mean(dim=-1)
        return {"logits": out["logits"], "aux": aux}


class SpikingMLPClassifier(nn.Module):
    def __init__(self, in_size=16, hidden=(256, 512, 256), n_classes=10, device=None):
        super().__init__()
        self.in_size = int(in_size)
        self.h1, self.h2, self.h3 = [int(x) for x in hidden]
        self.n_classes = int(n_classes)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc1 = nn.Linear(self.in_size, self.h1)
        self.fc2 = nn.Linear(self.h1, self.h2)
        self.fc3 = nn.Linear(self.h2, self.h3)
        self.fc4 = nn.Linear(self.h3, self.n_classes)

    def forward(self, X_seq, mem_thr=0.3, decay=0.9, lens=0.2):
        B, T, _ = X_seq.shape
        h1_mem = torch.zeros(B, self.h1, device=X_seq.device)
        h2_mem = torch.zeros(B, self.h2, device=X_seq.device)
        h3_mem = torch.zeros(B, self.h3, device=X_seq.device)
        h1_spk = torch.zeros_like(h1_mem)
        h2_spk = torch.zeros_like(h2_mem)
        h3_spk = torch.zeros_like(h3_mem)
        out_sum = torch.zeros(B, self.n_classes, device=X_seq.device)
        for t in range(T):
            x_t = X_seq[:, t, :]
            h1_mem = h1_mem * decay * (1.0 - h1_spk) + self.fc1(x_t)
            h1_spk = act_fun(h1_mem, mem_thr, lens)
            h2_mem = h2_mem * decay * (1.0 - h2_spk) + self.fc2(h1_spk)
            h2_spk = act_fun(h2_mem, mem_thr, lens)
            h3_mem = h3_mem * decay * (1.0 - h3_spk) + self.fc3(h2_spk)
            h3_spk = act_fun(h3_mem, mem_thr, lens)
            out_sum = out_sum + self.fc4(h3_spk)
        return out_sum / T


@torch.no_grad()
def accuracy(logits, y):
    pred = torch.argmax(logits, dim=1)
    return (pred == y).float().mean().item()


@torch.no_grad()
def batch_diagnostics(aux: Dict[str, Any]) -> Dict[str, float]:
    diag = {}
    if not aux:
        return diag
    tensor_keys = ["spike_mean", "dead_ratio", "state_norm", "rate_state_norm", "temporal_norm", "ze_norm", "zp_norm", "zs_norm", "gate_mean", "gate_min", "gate_max", "gate_active_frac"]
    for k in tensor_keys:
        v = aux.get(k)
        if isinstance(v, torch.Tensor):
            diag[k] = float(v.detach().mean().cpu())
    bw = aux.get("branch_weights")
    if isinstance(bw, torch.Tensor) and bw.numel() == 3:
        diag["branch_w_e"] = float(bw[0].detach().cpu())
        diag["branch_w_p"] = float(bw[1].detach().cpu())
        diag["branch_w_s"] = float(bw[2].detach().cpu())
    return diag


def train_epoch(model, loader, optimizer, device, log_every=200, return_diagnostics: bool = False, **forward_kwargs):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    diag_sums: Dict[str, float] = {}
    diag_count = 0
    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        model_out = model(x, return_aux=return_diagnostics, **forward_kwargs)
        logits, aux = unpack_model_output(model_out)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs
        if return_diagnostics and aux:
            d = batch_diagnostics(aux)
            for k, v in d.items():
                diag_sums[k] = diag_sums.get(k, 0.0) + float(v)
            diag_count += 1
        if log_every and (step + 1) % log_every == 0:
            print(f"  step {step+1:04d}: loss={loss.item():.4f}")
    avg_loss = total_loss / max(n, 1)
    if not return_diagnostics:
        return avg_loss, _
    avg_diag = {k: v / max(diag_count, 1) for k, v in diag_sums.items()}
    return avg_loss, _, avg_diag


@torch.no_grad()
def eval_epoch(model, loader, device, return_diagnostics: bool = False, **forward_kwargs):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    diag_sums: Dict[str, float] = {}
    diag_count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        model_out = model(x, return_aux=return_diagnostics, **forward_kwargs)
        logits, aux = unpack_model_output(model_out)
        loss = F.cross_entropy(logits, y)
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, y) * bs
        n += bs
        if return_diagnostics and aux:
            d = batch_diagnostics(aux)
            for k, v in d.items():
                diag_sums[k] = diag_sums.get(k, 0.0) + float(v)
            diag_count += 1
    avg_loss = total_loss / max(n, 1)
    avg_acc = total_acc / max(n, 1)
    if not return_diagnostics:
        return avg_loss, avg_acc
    avg_diag = {k: v / max(diag_count, 1) for k, v in diag_sums.items()}
    return avg_loss, avg_acc, avg_diag
