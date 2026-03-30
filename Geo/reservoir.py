import math
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .topology import small_world_connectivity
from .utils import print_summary
from .training import STDP


class Reservoir():
  def __init__(self, cube_shape=(10,10,10), inputs=None, coordinates=None, mapping=None,
               c=0.4, l=0.169, c_in=0.9, l_in=1.2, use_mps=False):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available() and use_mps is True:
      self.device = torch.device("mps:0")

    if coordinates is None:
      self.n_neurons = math.prod(cube_shape)
      x, y, z = torch.meshgrid(
          torch.linspace(0, 1, cube_shape[0]),
          torch.linspace(0, 1, cube_shape[1]),
          torch.linspace(0, 1, cube_shape[2]),
          indexing='xy'
      )
      self.pos = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1).to(self.device)
    else:
      self.n_neurons = coordinates.shape[0]
      self.pos = coordinates.to(self.device)

    dist = torch.cdist(self.pos, self.pos)
    conn_mat = small_world_connectivity(dist, c=c, l=l) / 100.0

    inh_n = torch.randint(self.n_neurons, size=(int(self.n_neurons*0.2),), device=self.device)
    conn_mat[:, inh_n] = -conn_mat[:, inh_n]

    if mapping is None:
      if inputs is None:
        raise ValueError("")
      input_conn = torch.where(
          torch.rand(self.n_neurons, inputs, device=self.device) > 0.95,
          torch.ones(self.n_neurons, inputs, device=self.device),
          torch.zeros(self.n_neurons, inputs, device=self.device)
      ) / 50.0
    else:
      dist_in = torch.cdist(self.pos, mapping.to(self.device), p=2)
      input_conn = small_world_connectivity(dist_in, c=c_in, l=l_in) / 50.0

    self.w_latent = conn_mat.to(self.device)
    self.w_in = input_conn.to(self.device)

  def simulate(self, X, mem_thr=0.1, refractory_period=5, train=True,
               learning_rule=STDP(), verbose=True):
    if train is True and learning_rule is None:
      raise Exception("")

    self.batch_size, self.n_time, self.n_features = X.shape
    spike_rec = torch.zeros(self.batch_size, self.n_time, self.n_neurons, device=self.device)

    if train is True:
      learning_rule.setup(self.device, self.n_neurons)

    for s in tqdm(range(X.shape[0]), disable = not verbose):
      spike_latent = torch.zeros(self.n_neurons, device=self.device)
      mem_poten = torch.zeros(self.n_neurons, device=self.device)
      refrac = torch.ones(self.n_neurons, device=self.device)
      refrac_count = torch.zeros(self.n_neurons, device=self.device)
      spike_times = torch.zeros(self.n_neurons, device=self.device)

      if train is True:
        learning_rule.per_sample(s)

      for k in range(self.n_time):
        spike_in = X[s,k,:].to(self.device)

        refrac[refrac_count < 1] = 1

        I = torch.sum(self.w_in*spike_in, axis=1) + torch.sum(self.w_latent*spike_latent, axis=1)
        mem_poten = mem_poten*torch.exp(torch.tensor(-(1/40), device=self.device))*(1-spike_latent) + (refrac*I)

        spike_latent = (mem_poten >= mem_thr).to(mem_poten.dtype)

        refrac[mem_poten >= mem_thr] = 0
        refrac_count[mem_poten >= mem_thr] = refractory_period
        refrac_count = refrac_count-1

        if train is True:
          learning_rule.per_time_slice(s, k)
          pre_updates, pos_updates = learning_rule.train(k-spike_times, self.w_latent, spike_latent)
          self.w_latent += pre_updates.to(self.device)
          self.w_latent += pos_updates.to(self.device)
          learning_rule.reset()

        spike_times[mem_poten >= mem_thr] = k
        spike_rec[s,k,:] = spike_latent

    return spike_rec

  def summary(self):
    res_info = [["Neurons", str(self.n_neurons)],
                ["Reservoir connections", str((self.w_latent != 0).sum().item())],
                ["Input connections", str((self.w_in != 0).sum().item())],
                ["Device", str(self.device)]]
    print_summary(res_info)


class _SurrogateSpike(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, alpha: float):
    ctx.save_for_backward(x)
    ctx.alpha = alpha
    return (x > 0).to(x.dtype)

  @staticmethod
  def backward(ctx, grad_output):
    (x,) = ctx.saved_tensors
    alpha = ctx.alpha
    grad = grad_output / (alpha*torch.abs(x) + 1.0)**2
    return grad, None


class _SurrogateSpikeBoxcar(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, lens: float):
    ctx.save_for_backward(x)
    ctx.lens = float(lens)
    return (x > 0).to(x.dtype)

  @staticmethod
  def backward(ctx, grad_output):
    (x,) = ctx.saved_tensors
    lens = ctx.lens
    mask = (torch.abs(x) < lens).to(x.dtype)
    return grad_output * mask, None


class DifferentiableReservoir(nn.Module):

  def __init__(
      self,
      cube_shape=(10,10,10),
      inputs: int = 784,
      coordinates=None,
      mapping=None,
      c=0.4, l=0.169,
      c_in=0.9, l_in=1.2,
      inh_frac=0.2,
      w_scale_latent=0.01,
      w_scale_in=0.02,
      train_w_in=False,
      train_w_latent=False,
      max_tasks: int = 32,
      task_gate_floor: float = 0.02,
      gate_k_frac: float = 0.25,
      gate_init_scale: float = 6.0,
      history_bias_strength: float = 6.0,
      use_mps=False,
      device=None,
  ):
    super().__init__()
    if device is None:
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      if torch.backends.mps.is_available() and use_mps:
        device = torch.device("mps:0")
    self.device = device

    if coordinates is None:
      self.n_neurons = math.prod(cube_shape)
      x, y, z = torch.meshgrid(
          torch.linspace(0, 1, cube_shape[0]),
          torch.linspace(0, 1, cube_shape[1]),
          torch.linspace(0, 1, cube_shape[2]),
          indexing='xy'
      )
      pos = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)
    else:
      self.n_neurons = coordinates.shape[0]
      pos = coordinates
    self.register_buffer("pos", pos.to(self.device))

    dist = torch.cdist(self.pos, self.pos)
    conn_mat = small_world_connectivity(dist, c=c, l=l)

    conn_mat = (conn_mat * w_scale_latent).to(torch.float32)
    inh_n = torch.randint(self.n_neurons, size=(int(self.n_neurons * inh_frac),), device=self.device)
    conn_mat[:, inh_n] = -conn_mat[:, inh_n]

    if mapping is None:
      input_conn = torch.where(
          torch.rand(self.n_neurons, inputs, device=self.device) > 0.95,
          torch.ones(self.n_neurons, inputs, device=self.device),
          torch.zeros(self.n_neurons, inputs, device=self.device)
      ).to(torch.float32)
      input_conn = input_conn * w_scale_in
    else:
      dist_in = torch.cdist(self.pos, mapping.to(self.device), p=2)
      input_conn = small_world_connectivity(dist_in, c=c_in, l=l_in).to(torch.float32)
      input_conn = input_conn * w_scale_in

    self.register_buffer("mask_latent", (conn_mat != 0).to(torch.float32))
    self.register_buffer("mask_in", (input_conn != 0).to(torch.float32))

    self.w_latent = nn.Parameter(conn_mat, requires_grad=bool(train_w_latent))
    self.w_in = nn.Parameter(input_conn, requires_grad=bool(train_w_in))

    self.max_tasks = int(max_tasks)
    self.task_gate_floor = float(task_gate_floor)
    self.gate_k_frac = float(gate_k_frac)
    self.gate_init_scale = float(gate_init_scale)
    self.history_bias_strength = float(history_bias_strength)

    init_logits = self._init_task_gate_logits(max_tasks=self.max_tasks, scale=self.gate_init_scale)
    self.task_gate_logits = nn.Parameter(init_logits)

    self.register_buffer("frozen_task_gates", torch.zeros(self.max_tasks, self.n_neurons, dtype=torch.float32, device=self.device))
    self.register_buffer("frozen_task_active", torch.zeros(self.max_tasks, dtype=torch.float32, device=self.device))

    smooth_adj = ((self.mask_latent + self.mask_latent.T) > 0).to(torch.float32)
    smooth_adj.fill_diagonal_(0.0)
    smooth_deg = smooth_adj.sum(dim=-1)
    smooth_lap = torch.diag(smooth_deg) - smooth_adj
    self.register_buffer("smooth_adj", smooth_adj)
    self.register_buffer("smooth_laplacian", smooth_lap)

  def _init_task_gate_logits(self, max_tasks: int, scale: float) -> torch.Tensor:
    gen = torch.Generator(device=self.device)
    gen.manual_seed(12345)
    anchors = torch.rand((int(max_tasks), 3), generator=gen, device=self.device, dtype=self.pos.dtype)
    dist = torch.cdist(anchors, self.pos)
    dist = dist / (dist.mean(dim=-1, keepdim=True) + 1e-6)
    logits = -float(scale) * dist
    noise = torch.randn(logits.shape, generator=gen, device=self.device, dtype=logits.dtype)
    logits = logits + 0.05 * noise
    return logits

  def reset_state(self, batch: int):
    mem = torch.zeros(batch, self.n_neurons, device=self.device)
    spk = torch.zeros(batch, self.n_neurons, device=self.device)
    return mem, spk

  def _expand_task_id(self, task_id: Optional[Union[torch.Tensor, int]], batch: int) -> Optional[torch.Tensor]:
    if task_id is None:
      return None
    if not torch.is_tensor(task_id):
      task_id = torch.tensor([int(task_id)], device=self.device, dtype=torch.long)
    task_id = task_id.to(device=self.device, dtype=torch.long).view(-1)
    if task_id.numel() == 1 and batch > 1:
      task_id = task_id.expand(batch)
    if task_id.numel() != batch:
      raise ValueError(f"task_id must have 1 or batch elements, got {task_id.numel()} for batch={batch}")
    task_id = torch.clamp(task_id, min=0, max=self.max_tasks - 1)
    return task_id

  def _hard_topk_gate(self, logits: torch.Tensor) -> torch.Tensor:
    k = max(1, min(self.n_neurons, int(round(self.gate_k_frac * self.n_neurons))))
    topk_idx = torch.topk(logits, k=k, dim=-1).indices
    hard = torch.zeros_like(logits)
    hard.scatter_(dim=-1, index=topk_idx, src=torch.ones_like(topk_idx, dtype=logits.dtype))
    return hard

  def _history_occupancy(self, task_id: torch.Tensor) -> torch.Tensor:
    batch = int(task_id.numel())
    active_ids = torch.nonzero(self.frozen_task_active > 0.5, as_tuple=False).view(-1)
    if active_ids.numel() == 0:
      return torch.zeros(batch, self.n_neurons, device=self.device, dtype=torch.float32)

    active_gates = self.frozen_task_gates[active_ids]  # [A, N]
    neq = active_ids.unsqueeze(0) != task_id.unsqueeze(1)  # [B, A]
    if active_gates.numel() == 0:
      return torch.zeros(batch, self.n_neurons, device=self.device, dtype=torch.float32)
    occ = (neq.to(active_gates.dtype).unsqueeze(-1) * active_gates.unsqueeze(0)).amax(dim=1)
    return occ

  @torch.no_grad()
  def freeze_task_gate(self, task_id: int) -> torch.Tensor:
    task_idx = int(task_id)
    task_tensor = torch.tensor([task_idx], device=self.device, dtype=torch.long)
    logits = self.task_gate_logits[task_tensor]
    occ = self._history_occupancy(task_tensor)
    eff_logits = logits - self.history_bias_strength * occ
    hard = self._hard_topk_gate(eff_logits)
    self.frozen_task_gates[task_idx].copy_(hard.squeeze(0))
    self.frozen_task_active[task_idx] = 1.0
    return self.frozen_task_gates[task_idx].clone()

  def _task_gate_raw(
      self,
      task_id: Optional[Union[torch.Tensor, int]],
      batch: Optional[int] = None,
      *,
      use_history_bias: bool = True,
      use_frozen_if_available: bool = True,
  ):
    if task_id is None:
      return None, None, None
    if batch is None:
      batch = int(task_id.numel()) if torch.is_tensor(task_id) else 1
    task_id = self._expand_task_id(task_id, batch=batch)

    raw = torch.zeros(batch, self.n_neurons, device=self.device, dtype=torch.float32)
    probs = torch.zeros_like(raw)
    logits_out = torch.zeros_like(raw)

    frozen_available = self.frozen_task_active[task_id] > 0.5
    if use_frozen_if_available and frozen_available.any():
      frozen = self.frozen_task_gates[task_id[frozen_available]]
      raw[frozen_available] = frozen
      probs[frozen_available] = frozen
      logits_out[frozen_available] = torch.where(
          frozen > 0,
          torch.full_like(frozen, 8.0),
          torch.full_like(frozen, -8.0),
      )

    learn_mask = ~frozen_available if use_frozen_if_available else torch.ones_like(frozen_available, dtype=torch.bool)
    if learn_mask.any():
      learn_ids = task_id[learn_mask]
      logits = self.task_gate_logits[learn_ids]
      if use_history_bias and self.history_bias_strength > 0:
        occ = self._history_occupancy(learn_ids)
        logits_eff = logits - self.history_bias_strength * occ
      else:
        logits_eff = logits
      probs_live = torch.sigmoid(logits_eff)
      hard_live = self._hard_topk_gate(logits_eff)
      if self.training:
        raw_live = hard_live + probs_live - probs_live.detach()
      else:
        raw_live = hard_live
      raw[learn_mask] = raw_live
      probs[learn_mask] = probs_live
      logits_out[learn_mask] = logits_eff

    return raw, probs, logits_out

  def task_gate(self, task_id: Optional[Union[torch.Tensor, int]], batch: Optional[int] = None) -> Optional[torch.Tensor]:
    raw_gate, _, _ = self._task_gate_raw(task_id=task_id, batch=batch)
    if raw_gate is None:
      return None
    if self.task_gate_floor > 0:
      return self.task_gate_floor + (1.0 - self.task_gate_floor) * raw_gate
    return raw_gate

  def task_edge_gate(self, task_id: Optional[Union[torch.Tensor, int]], batch: Optional[int] = None) -> Optional[torch.Tensor]:
    gate = self.task_gate(task_id=task_id, batch=batch)
    if gate is None:
      return None
    raw = gate
    if self.task_gate_floor > 0:
      raw = torch.clamp((gate - self.task_gate_floor) / max(1.0 - self.task_gate_floor, 1e-6), min=0.0, max=1.0)
    return raw.unsqueeze(-1) * raw.unsqueeze(-2)

  def gate_regularization(
      self,
      task_id: int,
      previous_task_ids: Optional[Sequence[int]] = None,
      sparsity_weight: float = 0.0,
      overlap_weight: float = 0.0,
      binary_weight: float = 0.0,
      smooth_weight: float = 0.0,
      eps: float = 1e-6,
  ):
    device = self.device
    raw, probs, _ = self._task_gate_raw(task_id=task_id, batch=1, use_history_bias=True, use_frozen_if_available=False)
    if raw is None or probs is None:
      zero = torch.tensor(0.0, device=device)
      return zero, {
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

    zero = torch.tensor(0.0, device=device)
    raw = raw.squeeze(0)
    probs = probs.squeeze(0)

    loss_sparse = probs.mean() if float(sparsity_weight) > 0 else zero
    loss_binary = (probs * (1.0 - probs)).mean() if float(binary_weight) > 0 else zero

    loss_smooth = zero
    if float(smooth_weight) > 0 and self.smooth_adj.sum() > 0:
      g = probs
      quad = (g @ self.smooth_laplacian @ g) / (self.smooth_adj.sum() + eps)
      loss_smooth = quad

    loss_overlap = zero
    mean_overlap = 0.0
    max_overlap = 0.0
    prev_ids = [int(t) for t in (previous_task_ids or []) if int(t) != int(task_id)]
    if float(overlap_weight) > 0 and len(prev_ids) > 0:
      prev_raw, _, _ = self._task_gate_raw(
          task_id=torch.tensor(prev_ids, device=device, dtype=torch.long),
          batch=len(prev_ids),
          use_history_bias=False,
          use_frozen_if_available=True,
      )
      curr_n = F.normalize(raw, p=2, dim=-1, eps=eps)
      prev_n = F.normalize(prev_raw.detach(), p=2, dim=-1, eps=eps)
      sims = prev_n @ curr_n
      loss_overlap = (sims ** 2).mean()
      mean_overlap = float(sims.detach().mean().cpu())
      max_overlap = float(sims.detach().max().cpu())

    loss = (
        float(sparsity_weight) * loss_sparse
        + float(overlap_weight) * loss_overlap
        + float(binary_weight) * loss_binary
        + float(smooth_weight) * loss_smooth
    )

    logs = {
        "loss_gate_sparse": float(loss_sparse.detach().cpu()),
        "loss_gate_overlap": float(loss_overlap.detach().cpu()),
        "loss_gate_binary": float(loss_binary.detach().cpu()),
        "loss_gate_smooth": float(loss_smooth.detach().cpu()),
        "gate_mean": float(probs.detach().mean().cpu()),
        "gate_min": float(probs.detach().min().cpu()),
        "gate_max": float(probs.detach().max().cpu()),
        "gate_active_frac": float(raw.detach().mean().cpu()),
        "gate_overlap_mean": mean_overlap,
        "gate_overlap_max": max_overlap,
      }
    return loss, logs

  def step(
      self,
      x_t: torch.Tensor,
      mem: torch.Tensor,
      spk: torch.Tensor,
      mem_thr: float = 0.3,
      decay: float = 0.9,
      alpha: float = 2.0,
      lens: float = 0.2,
      task_id: Optional[Union[torch.Tensor, int]] = None,
  ):
    x_t = x_t.to(self.device)
    batch = x_t.size(0)

    w_in_eff = self.w_in * self.mask_in
    w_lat_eff = self.w_latent * self.mask_latent

    gate = self.task_gate(task_id=task_id, batch=batch)

    I_in = x_t @ w_in_eff.T
    if gate is not None:
      I_in = I_in * gate

    spk_src = spk if gate is None else spk * gate
    I_rec = spk_src @ w_lat_eff.T
    if gate is not None:
      I_rec = I_rec * gate

    I = I_in + I_rec
    mem = mem * decay * (1.0 - spk) + I
    if gate is not None:
      mem = mem * gate

    if lens is not None:
      spk = _SurrogateSpikeBoxcar.apply(mem - mem_thr, float(lens))
    else:
      spk = _SurrogateSpike.apply(mem - mem_thr, float(alpha))
    if gate is not None:
      spk = spk * gate
    return mem, spk

  def forward(
      self,
      X: torch.Tensor,
      mem_thr: float = 0.3,
      tau_mem: float = 20.0,
      decay: Optional[float] = None,
      alpha: float = 2.0,
      lens: float = 0.2,
      return_mem: bool = False,
      task_id: Optional[Union[torch.Tensor, int]] = None,
  ):
    X = X.to(self.device)
    batch, time, _ = X.shape

    if decay is None:
      decay = torch.exp(torch.tensor(-1.0 / tau_mem, device=self.device)).item()

    mem, spk = self.reset_state(batch)
    spikes = []
    mems = [] if return_mem else None

    task_id = self._expand_task_id(task_id, batch=batch) if task_id is not None else None

    for t in range(time):
      mem, spk = self.step(
          X[:, t, :],
          mem,
          spk,
          mem_thr=mem_thr,
          decay=float(decay),
          alpha=alpha,
          lens=lens,
          task_id=task_id,
      )
      spikes.append(spk)
      if return_mem:
        mems.append(mem)

    spike_rec = torch.stack(spikes, dim=1)
    if return_mem:
      return spike_rec, torch.stack(mems, dim=1)
    return spike_rec

  def summary(self):
    rows = [
        ["Neurons", str(self.n_neurons)],
        ["Reservoir connections (nonzero)", str((self.w_latent.detach() != 0).sum().item())],
        ["Input connections (nonzero)", str((self.w_in.detach() != 0).sum().item())],
        ["Train w_latent", str(self.w_latent.requires_grad)],
        ["Train w_in", str(self.w_in.requires_grad)],
        ["Max tasks", str(self.max_tasks)],
        ["Task gate floor", f"{self.task_gate_floor:.3f}"],
        ["Task gate active frac", f"{self.gate_k_frac:.3f}"],
        ["History bias strength", f"{self.history_bias_strength:.3f}"],
        ["Device", str(self.device)],
    ]
    print_summary(rows, title="GeoEngram Summary")
