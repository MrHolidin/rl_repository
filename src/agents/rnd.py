"""Random Network Distillation (RND) intrinsic-motivation exploration for bglike.

Self-contained and host-side. Novelty is measured over a FIXED featurization of
the agent's OWN board — a per-minion-type ``(n_normal, n_golden)`` count vector —
so the bonus rewards novel board *compositions* (combo discovery), not the
noisy shop/opponent randomness that would hijack a full-observation detector
(the "noisy-TV" failure mode).

Reference: Burda et al., "Exploration by Random Network Distillation"
(arXiv:1810.12894). Defaults follow the paper + the CleanRL ``ppo_rnd`` reference
(obs whitening + clip[-5,5], intrinsic-return normalization, predictor trained on
a random ``update_proportion`` of samples, target frozen).

The target/predictor are MLPs with a nonlinear hidden layer ON PURPOSE: a purely
linear detector is additively separable per card and blind to co-occurrence, so
a never-seen A+B board would carry no novelty. The hidden GELU layer makes the
target's response to a combination not derivable from its single-card responses,
which is exactly what surfaces novel *combos*.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from src.envs.bglike.actions import BOARD_SIZE
from src.envs.bglike.obs import BGLIKE_GLOBAL_DIM
from src.envs.minibg.obs import (
    CARD_IDX_OFFSET,
    GOLDEN_OFFSET,
    PRESENCE_OFFSET,
    SLOT_DIM,
)

# The own-board region sits right after the flat globals in the v3-head, which is
# a strict prefix of obs_v5 / obs_v5_heroes (and of the +identity tail), so this
# offset is stable across all v11 obs variants.
_OWN_OFFSET = int(BGLIKE_GLOBAL_DIM)
_OWN_LEN = int(BOARD_SIZE)


def own_board_counts(obs: torch.Tensor, num_pool_indices: int) -> torch.Tensor:
    """``(B, obs_dim) -> (B, num_pool_indices * 2)``: per-card ``(n_normal, n_golden)``.

    Reads card_idx / presence / golden straight from the own-board slots using the
    single-source slot offsets. ``card_idx == 0`` is the empty/pad slot and is
    dropped, so column ``c`` of each half is minion-pool index ``c + 1``.
    """
    B = obs.shape[0]
    own = obs[:, _OWN_OFFSET : _OWN_OFFSET + _OWN_LEN * SLOT_DIM].view(B, _OWN_LEN, SLOT_DIM)
    card_idx = own[:, :, CARD_IDX_OFFSET].round().long().clamp_(0, num_pool_indices)
    occupied = (own[:, :, PRESENCE_OFFSET] > 0.5) & (card_idx > 0)
    is_golden = occupied & (own[:, :, GOLDEN_OFFSET] > 0.5)
    is_normal = occupied & ~is_golden

    width = num_pool_indices + 1  # column 0 == empty/pad, dropped below
    normal = obs.new_zeros(B, width)
    golden = obs.new_zeros(B, width)
    normal.scatter_add_(1, card_idx, is_normal.to(obs.dtype))
    golden.scatter_add_(1, card_idx, is_golden.to(obs.dtype))
    return torch.cat([normal[:, 1:], golden[:, 1:]], dim=1)


def discounted_forward(rewards: Sequence[float], gamma: float) -> np.ndarray:
    """Forward-discounted accumulation ``g_t = r_t + gamma * g_{t-1}`` (CleanRL's
    ``RewardForwardFilter``). The variance of these values is what the intrinsic
    reward is normalized against (it tracks the scale of the intrinsic *return*,
    not of a single reward)."""
    out = np.empty(len(rewards), dtype=np.float64)
    acc = 0.0
    for i, r in enumerate(rewards):
        acc = float(r) + gamma * acc
        out[i] = acc
    return out


def _mlp(sizes: Sequence[int], *, gain: Optional[float] = None) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        lin = nn.Linear(sizes[i], sizes[i + 1])
        if gain is not None:
            # Orthogonal init with a HIGH gain pushes GELU into its nonlinear
            # regime, so the random target's higher-order (combo / triple)
            # interaction terms are strong. A novel combination of familiar cards
            # then reads far more novel than a near-linear shallow MLP leaves it
            # (probe: held-out combo went from ~2% of a brand-new card to ~15-26%,
            # triples too) — without explicit pairwise features. Too high a gain,
            # though, stops the predictor from fitting visited boards (novelty
            # everywhere = noise); ~2.5-3.0 is the sweet spot.
            nn.init.orthogonal_(lin.weight, gain=gain)
            nn.init.zeros_(lin.bias)
        layers.append(lin)
        if i < len(sizes) - 2:  # nonlinearity on every layer but the last
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class RNDModel(nn.Module):
    """Frozen random target + trained predictor over the board-count features,
    plus the persistent normalization statistics (obs whitening + intrinsic-return
    scale). All state — nets and running stats — lives in the module's
    ``state_dict`` so checkpointing is a single ``rnd.state_dict()``."""

    def __init__(
        self,
        num_pool_indices: int,
        *,
        embed_dim: int = 128,
        target_hidden: int = 256,
        predictor_hidden: int = 256,
        predictor_layers: int = 2,
        target_layers: int = 1,
        init_gain: float = 0.0,
        obs_clip: float = 5.0,
        epsilon: float = 1e-4,
    ) -> None:
        super().__init__()
        self.num_pool_indices = int(num_pool_indices)
        self.in_dim = self.num_pool_indices * 2
        self.obs_clip = float(obs_clip)
        self._eps = float(epsilon)

        gain = float(init_gain) if init_gain and init_gain > 0 else None
        # Target: ``target_layers`` nonlinear hidden layers; depth + a high init
        # gain make the combo/triple interaction terms strong. Frozen forever.
        tgt_sizes = [self.in_dim] + [target_hidden] * max(1, int(target_layers)) + [embed_dim]
        self.target = _mlp(tgt_sizes, gain=gain)
        # Predictor: deeper, so it can fit the target on visited boards while
        # leaving error on the unvisited combinations.
        pred_sizes = [self.in_dim] + [predictor_hidden] * int(predictor_layers) + [embed_dim]
        self.predictor = _mlp(pred_sizes, gain=gain)
        for p in self.target.parameters():
            p.requires_grad_(False)

        # Per-dim obs normalization (Welford running mean/var) and scalar
        # intrinsic-return normalization. float64 for numerically stable updates.
        self.register_buffer("obs_mean", torch.zeros(self.in_dim, dtype=torch.float64))
        self.register_buffer("obs_var", torch.ones(self.in_dim, dtype=torch.float64))
        self.register_buffer("obs_count", torch.tensor(float(epsilon), dtype=torch.float64))
        self.register_buffer("ret_mean", torch.zeros((), dtype=torch.float64))
        self.register_buffer("ret_var", torch.ones((), dtype=torch.float64))
        self.register_buffer("ret_count", torch.tensor(float(epsilon), dtype=torch.float64))

    # -- featurization + obs normalization ------------------------------------
    def featurize(self, obs: torch.Tensor) -> torch.Tensor:
        return own_board_counts(obs, self.num_pool_indices)

    def _normalize_obs(self, feat: torch.Tensor) -> torch.Tensor:
        mean = self.obs_mean.to(feat.dtype)
        std = torch.sqrt(self.obs_var.to(feat.dtype) + 1e-8)
        return ((feat - mean) / std).clamp(-self.obs_clip, self.obs_clip)

    @torch.no_grad()
    def update_obs_rms(self, feat: torch.Tensor) -> None:
        x = feat.to(torch.float64)
        n = x.shape[0]
        if n == 0:
            return
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        self._welford(self.obs_mean, self.obs_var, self.obs_count, batch_mean, batch_var, n)

    @torch.no_grad()
    def update_ret_rms(self, returns: np.ndarray) -> None:
        if returns.size == 0:
            return
        t = torch.as_tensor(returns, dtype=torch.float64, device=self.ret_var.device)
        self._welford(
            self.ret_mean, self.ret_var, self.ret_count,
            t.mean(), t.var(unbiased=False), int(t.numel()),
        )

    @staticmethod
    def _welford(mean, var, count, batch_mean, batch_var, n) -> None:
        delta = batch_mean - mean
        tot = count + n
        mean += delta * n / tot
        m_a = var * count
        m_b = batch_var * n
        var.copy_((m_a + m_b + delta * delta * count * n / tot) / tot)
        count.copy_(tot)

    def ret_std(self) -> float:
        return float(torch.sqrt(self.ret_var + 1e-8).item())

    @torch.no_grad()
    def reset_predictor(self) -> None:
        """Re-randomize the predictor and reset the intrinsic-return scale.

        A single predictor converges to a low novelty floor on the visited
        manifold and stays there — its decay IS its learning, so it cannot be
        slowed enough to keep discriminating into late training without becoming
        uninformative. Periodically re-randomizing it re-injects a fresh wave:
        right after a reset it errs everywhere, then re-learns the (current)
        common boards first, so discrimination over *currently rare* boards comes
        back — including boards the policy only started visiting recently. The
        target stays frozen (the reference must not move). ret-rms is reset so the
        reward re-normalizes to the post-reset error scale (obs-rms is kept)."""
        for m in self.predictor.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        self.ret_mean.zero_()
        self.ret_var.fill_(1.0)
        self.ret_count.fill_(self._eps)

    # -- novelty + predictor loss ---------------------------------------------
    @torch.no_grad()
    def novelty(self, feat: torch.Tensor) -> torch.Tensor:
        """Per-row prediction error on already-collected boards (the raw intrinsic
        reward, before intrinsic-return normalization). Uses the CURRENT obs-rms."""
        x = self._normalize_obs(feat)
        err = (self.predictor(x) - self.target(x)).pow(2).mean(dim=-1)
        return err

    def predictor_loss(self, feat: torch.Tensor) -> torch.Tensor:
        """Per-row MSE for training the predictor (caller applies the 0.25 mask).
        obs is detached from the rms (no grad to the stats); the target is frozen."""
        x = self._normalize_obs(feat).detach()
        return (self.predictor(x) - self.target(x)).pow(2).mean(dim=-1)


__all__ = [
    "RNDModel",
    "own_board_counts",
    "discounted_forward",
]
