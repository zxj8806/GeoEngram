from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F


class EpisodicMemoryBuffer:
    def __init__(self, per_class: int = 20, device: str = "cpu") -> None:
        self.per_class = int(per_class)
        self.device = torch.device(device)
        self._buffers: Dict[Tuple[int, int], deque] = defaultdict(lambda: deque(maxlen=self.per_class))

    def __len__(self) -> int:
        return sum(len(v) for v in self._buffers.values())

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    def task_class_keys(self) -> List[Tuple[int, int]]:
        return sorted(self._buffers.keys())

    def _pack_item(self, x: torch.Tensor, y: int, task_id: int, aux: Dict[str, Any]) -> Dict[str, Any]:
        item: Dict[str, Any] = {"x": x.detach().cpu(), "y": int(y), "task_id": int(task_id)}
        keep_keys = [
            "teacher_logits",
            "state",
            "engram_state",
            "rate_state",
            "route_signature",
            "temporal_signature",
            "ze",
            "zp",
            "zs",
            "de2",
            "dp2",
            "ds2",
            "branch_weights",
        ]
        for k in keep_keys:
            v = aux.get(k)
            if isinstance(v, torch.Tensor):
                item[k] = v.detach().cpu()
        return item

    def add_batch(self, x: torch.Tensor, y: torch.Tensor, aux: Dict[str, Any], task_id: int) -> None:
        bs = x.size(0)
        for i in range(bs):
            yi = int(y[i].item())
            aux_i = {}
            for k, v in aux.items():
                if not isinstance(v, torch.Tensor):
                    continue
                if v.dim() == 0:
                    aux_i[k] = v
                elif v.size(0) == bs:
                    aux_i[k] = v[i]
                else:
                    aux_i[k] = v
            self._buffers[(int(task_id), yi)].append(self._pack_item(x[i], yi, int(task_id), aux_i))

    def _all_group_items(self) -> List[Tuple[Tuple[int, int], List[Dict[str, Any]]]]:
        return [(k, list(buf)) for k, buf in self._buffers.items() if len(buf) > 0]

    def sample(self, batch_size: int, device: Optional[torch.device] = None) -> Optional[Dict[str, Any]]:
        if self.is_empty:
            return None
        if device is None:
            device = self.device
        groups = self._all_group_items()
        if len(groups) == 0:
            return None
        n = min(int(batch_size), len(self))
        picked: List[Dict[str, Any]] = []
        group_order = torch.randperm(len(groups)).tolist()
        group_ptr = 0
        while len(picked) < n:
            if group_ptr >= len(group_order):
                group_order = torch.randperm(len(groups)).tolist()
                group_ptr = 0
            gidx = group_order[group_ptr]
            group_ptr += 1
            _, items = groups[gidx]
            ridx = torch.randint(low=0, high=len(items), size=(1,)).item()
            picked.append(items[ridx])
        out: Dict[str, Any] = {
            "x": torch.stack([it["x"] for it in picked], dim=0).to(device),
            "y": torch.tensor([it["y"] for it in picked], dtype=torch.long, device=device),
            "task_id": torch.tensor([it["task_id"] for it in picked], dtype=torch.long, device=device),
        }
        optional_keys: Iterable[str] = [
            "teacher_logits",
            "state",
            "engram_state",
            "rate_state",
            "route_signature",
            "temporal_signature",
            "ze",
            "zp",
            "zs",
            "de2",
            "dp2",
            "ds2",
            "branch_weights",
        ]
        for k in optional_keys:
            vals = [it[k] for it in picked if k in it]
            if len(vals) == len(picked) and len(vals) > 0:
                out[k] = torch.stack(vals, dim=0).to(device)
        return out

    def all_items(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for _, buf in self._buffers.items():
            items.extend(list(buf))
        return items

    def stacked(self, key: str, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        vals = [it[key] for it in self.all_items() if key in it]
        if len(vals) == 0:
            return None
        out = torch.stack(vals, dim=0)
        if device is not None:
            out = out.to(device)
        return out

    def labels_tensor(self, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        items = self.all_items()
        if len(items) == 0:
            return None
        out = torch.tensor([it["y"] for it in items], dtype=torch.long)
        if device is not None:
            out = out.to(device)
        return out

    def task_ids_tensor(self, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        items = self.all_items()
        if len(items) == 0:
            return None
        out = torch.tensor([it["task_id"] for it in items], dtype=torch.long)
        if device is not None:
            out = out.to(device)
        return out

    def _centroid_from_items(
        self,
        items: List[Dict[str, Any]],
        key: str,
        *,
        normalize: bool = False,
        device: Optional[torch.device] = None,
        eps: float = 1e-6,
    ) -> Optional[torch.Tensor]:
        vals = [it[key] for it in items if key in it]
        if len(vals) == 0:
            return None
        c = torch.stack(vals, dim=0).mean(dim=0)
        if normalize:
            c = F.normalize(c, p=2, dim=-1, eps=eps)
        if device is not None:
            c = c.to(device)
        return c

    def task_class_route_centroids(self, device: Optional[torch.device] = None) -> Dict[Tuple[int, int], torch.Tensor]:
        out: Dict[Tuple[int, int], torch.Tensor] = {}
        for key, buf in self._buffers.items():
            centroid = self._centroid_from_items(list(buf), "route_signature", normalize=True, device=device)
            if centroid is not None:
                out[key] = centroid
        return out

    def task_class_engram_bank(self, device: Optional[torch.device] = None) -> Dict[Tuple[int, int], Dict[str, torch.Tensor]]:
        bank: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        target_device = device if device is not None else self.device
        for (task_id, class_id), buf in self._buffers.items():
            items = list(buf)
            if len(items) == 0:
                continue
            entry: Dict[str, torch.Tensor] = {
                "y": torch.tensor(int(class_id), dtype=torch.long, device=target_device),
                "task_id": torch.tensor(int(task_id), dtype=torch.long, device=target_device),
                "support_count": torch.tensor(len(items), dtype=torch.long, device=target_device),
            }
            centroid_specs = {
                "engram_state": False,
                "state": False,
                "rate_state": False,
                "route_signature": True,
                "temporal_signature": True,
                "ze": True,
                "zp": False,
                "zs": False,
                "de2": False,
                "dp2": False,
                "ds2": False,
                "teacher_logits": False,
                "branch_weights": False,
            }
            for k, normalize in centroid_specs.items():
                c = self._centroid_from_items(items, k, normalize=normalize, device=target_device)
                if c is not None:
                    entry[k] = c
            if "engram_state" not in entry and "state" in entry:
                entry["engram_state"] = entry["state"]
            if "state" not in entry and "engram_state" in entry:
                entry["state"] = entry["engram_state"]
            bank[(int(task_id), int(class_id))] = entry
        return bank

    def sample_engram_bank(self, batch_size: int, device: Optional[torch.device] = None) -> Optional[Dict[str, torch.Tensor]]:
        bank = self.task_class_engram_bank(device=device)
        if len(bank) == 0:
            return None
        keys = sorted(bank.keys())
        n = min(max(int(batch_size), 1), len(keys))
        perm = torch.randperm(len(keys)).tolist()[:n]
        picked = [bank[keys[i]] for i in perm]

        out: Dict[str, torch.Tensor] = {
            "y": torch.stack([it["y"] for it in picked], dim=0),
            "task_id": torch.stack([it["task_id"] for it in picked], dim=0),
            "support_count": torch.stack([it["support_count"] for it in picked], dim=0),
        }
        optional_keys = [
            "engram_state",
            "state",
            "rate_state",
            "route_signature",
            "temporal_signature",
            "ze",
            "zp",
            "zs",
            "de2",
            "dp2",
            "ds2",
            "teacher_logits",
            "branch_weights",
        ]
        for k in optional_keys:
            vals = [it[k] for it in picked if k in it]
            if len(vals) == len(picked) and len(vals) > 0:
                out[k] = torch.stack(vals, dim=0)
        return out
