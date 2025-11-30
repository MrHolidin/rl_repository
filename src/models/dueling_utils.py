"""Utilities for dueling Q-network aggregation."""

from __future__ import annotations

from typing import Optional

import torch

__all__ = ["dueling_aggregate"]


def dueling_aggregate(
    value: torch.Tensor,
    advantage: torch.Tensor,
    legal_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Combine value and advantage streams while ignoring illegal actions in the mean.
    """
    if legal_mask is None:
        centered_advantage = advantage - advantage.mean(dim=1, keepdim=True)
        return value + centered_advantage

    mask = legal_mask
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)

    mask = mask.to(device=advantage.device, dtype=advantage.dtype)
    masked_advantage = advantage * mask
    legal_counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    mean_advantage = masked_advantage.sum(dim=1, keepdim=True) / legal_counts

    return value + (masked_advantage - mean_advantage)

