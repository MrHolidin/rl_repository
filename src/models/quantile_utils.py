"""Quantile Regression DQN utilities."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def quantile_huber_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    taus: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    """
    Quantile Huber loss for QR-DQN.

    Args:
        predictions: (B, N) predicted quantile values
        targets: (B, N) target values (same shape)
        taus: (N,) quantile fractions in (0, 1)
        kappa: Huber loss threshold

    Returns:
        Scalar loss
    """
    td = targets - predictions  # (B, N)
    abs_td = td.abs()
    huber = torch.where(abs_td <= kappa, 0.5 * td.pow(2), kappa * (abs_td - 0.5 * kappa))
    weight = (taus.view(1, -1) - (td < 0).float()).abs()
    return (weight * huber).mean()


def make_tau_hat(n_quantiles: int, device: torch.device) -> torch.Tensor:
    """Central quantile fractions: tau_i = (2i + 1) / (2N)."""
    i = torch.arange(n_quantiles, dtype=torch.float32, device=device)
    return (2 * i + 1) / (2 * n_quantiles)
