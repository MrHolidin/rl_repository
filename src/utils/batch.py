"""Typed batch for DQN training and device transfer."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch

_ArrayOrTensor = Union[np.ndarray, torch.Tensor]
RawBatch = Tuple[
    _ArrayOrTensor, _ArrayOrTensor, _ArrayOrTensor, _ArrayOrTensor,
    _ArrayOrTensor, _ArrayOrTensor, _ArrayOrTensor,
    Optional[np.ndarray],
    _ArrayOrTensor,
]


@dataclass
class Batch:
    """Batch of transitions on target device. Weights always present, shape (B, 1) for broadcast with (B, N) per-sample loss; indices only for PER."""

    obs: torch.Tensor
    act: torch.Tensor
    rew: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor
    legal: torch.Tensor
    next_legal: torch.Tensor
    weights: torch.Tensor  # (B, 1)
    indices: Optional[np.ndarray] = None


def to_device(raw: RawBatch, device: torch.device) -> Batch:
    """Convert raw 9-tuple from replay.sample() to Batch on device."""
    (
        obs, act, rew, next_obs, done, legal, next_legal,
        indices, weights,
    ) = raw

    use_pinned = device.type == "cuda"
    non_blocking = use_pinned

    def _to_tensor(arr: _ArrayOrTensor, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            return arr.to(device=device, dtype=dtype, non_blocking=non_blocking)
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(dtype)
        if use_pinned:
            t = t.pin_memory()
        return t.to(device, non_blocking=non_blocking)

    w = _to_tensor(weights, torch.float32)
    if w.dim() == 1:
        w = w.unsqueeze(-1)

    return Batch(
        obs=_to_tensor(obs, torch.float32),
        act=_to_tensor(act, torch.long),
        rew=_to_tensor(rew, torch.float32),
        next_obs=_to_tensor(next_obs, torch.float32),
        done=_to_tensor(done, torch.bool),
        legal=_to_tensor(legal, torch.bool),
        next_legal=_to_tensor(next_legal, torch.bool),
        weights=w,
        indices=indices,
    )
