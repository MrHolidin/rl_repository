"""Flat-vector actor-critic for PPO (BGLike / generic MLP policies)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlatMLPActorCritic(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_actions: int,
        hidden_size: int = 256,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.num_actions = int(num_actions)
        self.hidden_size = int(hidden_size)
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.policy_head = nn.Linear(self.hidden_size, self.num_actions)
        self.value_head = nn.Linear(self.hidden_size, 1)

    def get_constructor_kwargs(self) -> Dict[str, Any]:
        return {
            "input_size": self.input_size,
            "num_actions": self.num_actions,
            "hidden_size": self.hidden_size,
        }

    def forward(
        self,
        x: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if x.shape[1] != self.input_size:
            raise ValueError(f"Expected obs dim {self.input_size}, got {x.shape[1]}")
        t = F.relu(self.fc1(x))
        t = F.relu(self.fc2(t))
        logits = self.policy_head(t)
        values = self.value_head(t).squeeze(-1)
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, float("-inf"))
        return logits, values


__all__ = ["FlatMLPActorCritic"]
