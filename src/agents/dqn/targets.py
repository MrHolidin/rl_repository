from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from ...utils.batch import Batch
from .action_selection import action_scores, masked_argmax


@dataclass(frozen=True)
class TargetCfg:
    gamma: float
    n_step: int = 1
    target_q_clip: float | None = None


@dataclass(frozen=True)
class OptimizeCfg:
    grad_clip_norm: float
    value_reg_weight: float = 0.0


@dataclass
class TargetOut:
    target: torch.Tensor
    next_actions: torch.Tensor
    extras: dict[str, float]


def build_scalar_targets(
    batch: Batch,
    main_net: nn.Module,
    target_nets: Sequence[nn.Module],
    cfg: TargetCfg,
) -> TargetOut:
    with torch.no_grad():
        next_out_main = main_net(batch.next_obs, legal_mask=batch.next_legal)
        scores = action_scores(next_out_main)
        next_actions = masked_argmax(scores, batch.next_legal)

        qs = []
        for tn in target_nets:
            q = tn(batch.next_obs, legal_mask=batch.next_legal)
            q = q.gather(1, next_actions[:, None]).squeeze(1)
            qs.append(q)
        target_next_q = torch.stack(qs, dim=0).min(dim=0).values

        gamma_bootstrap = cfg.gamma ** cfg.n_step if cfg.n_step > 1 else cfg.gamma
        target = batch.rew + (1 - batch.done.float()) * gamma_bootstrap * target_next_q

        extras: dict[str, float] = {}
        if cfg.target_q_clip is not None:
            before = target.clone()
            target = torch.clamp(target, -cfg.target_q_clip, cfg.target_q_clip)
            extras["target_clipped_frac_elem"] = (before != target).float().mean().item()
        else:
            extras["target_clipped_frac_elem"] = 0.0

    return TargetOut(target=target, next_actions=next_actions, extras=extras)


def build_quantile_targets(
    batch: Batch,
    main_net: nn.Module,
    target_net: nn.Module,
    cfg: TargetCfg,
) -> TargetOut:
    with torch.no_grad():
        next_out_main = main_net(batch.next_obs, legal_mask=batch.next_legal)
        next_actions = masked_argmax(action_scores(next_out_main), batch.next_legal)

        next_quantiles_target = target_net(batch.next_obs, legal_mask=batch.next_legal)
        N = next_quantiles_target.size(-1)
        target_quantiles = next_quantiles_target.gather(
            1,
            next_actions.unsqueeze(1).unsqueeze(2).expand(-1, 1, N),
        ).squeeze(1)

        gamma_bootstrap = cfg.gamma ** cfg.n_step if cfg.n_step > 1 else cfg.gamma
        target = batch.rew.unsqueeze(1) + (
            1 - batch.done.float().unsqueeze(1)
        ) * gamma_bootstrap * target_quantiles

        extras: dict[str, float] = {}
        if cfg.target_q_clip is not None:
            before = target.clone()
            target = torch.clamp(target, -cfg.target_q_clip, cfg.target_q_clip)
            extras["target_clipped_frac_elem"] = (before != target).float().mean().item()
        else:
            extras["target_clipped_frac_elem"] = 0.0

    return TargetOut(target=target, next_actions=next_actions, extras=extras)
