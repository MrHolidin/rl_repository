from typing import Sequence

import torch
import torch.nn.functional as F


def dqn_loss(
    preds: Sequence[torch.Tensor],
    target_q: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    w = weights.squeeze(-1) if weights.dim() > 1 else weights
    loss = torch.tensor(0.0, device=preds[0].device, dtype=preds[0].dtype)
    for p in preds:
        loss = loss + (F.smooth_l1_loss(p, target_q, reduction="none") * w).mean()
    td_abs = (target_q - preds[0]).detach().abs()
    return loss, td_abs


def qrdqn_loss(
    pred_quantiles: torch.Tensor,
    target_quantiles: torch.Tensor,
    tau: torch.Tensor,
    weights: torch.Tensor,
    kappa: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    td = target_quantiles - pred_quantiles
    abs_td = td.abs()
    huber = torch.where(abs_td <= kappa, 0.5 * td.pow(2), kappa * (abs_td - 0.5 * kappa))
    weight = (tau.view(1, -1) - (td < 0).float()).abs()
    per_sample = (weight * huber).mean(dim=1)
    w = weights.squeeze(-1) if weights.dim() > 1 else weights
    loss = (per_sample * w).mean()
    td_abs = (pred_quantiles.mean(dim=1) - target_quantiles.mean(dim=1)).abs().detach()
    return loss, td_abs
