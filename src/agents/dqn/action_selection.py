"""Action selection for DQN (scalar and distributional)."""

import torch


def action_scores(net_out: torch.Tensor) -> torch.Tensor:
    """Q-values for action selection: (B, A) from (B, A) or (B, A, N)."""
    return net_out.mean(-1) if net_out.dim() == 3 else net_out


def masked_argmax(scores: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Argmax over legal actions only. Returns (B,) long tensor."""
    scores = scores.masked_fill(~legal_mask, float("-inf"))
    no_legal = ~legal_mask.any(dim=1)
    scores[no_legal] = 0.0
    return scores.argmax(dim=1)
