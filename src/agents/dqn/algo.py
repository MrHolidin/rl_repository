from dataclasses import dataclass
from typing import Protocol, Sequence

import torch
import torch.nn as nn

from ...utils.batch import Batch
from ...models.quantile_utils import make_tau_hat
from .action_selection import action_scores
from .losses import dqn_loss, qrdqn_loss
from .targets import TargetCfg, TargetOut, build_scalar_targets, build_quantile_targets


@dataclass
class LossOut:
    loss: torch.Tensor
    td_abs: torch.Tensor
    extras: dict[str, float]


class Algo(Protocol):
    def infer_scores(self, net_out: torch.Tensor) -> torch.Tensor: ...
    def pred_for_actions(
        self, net_outs: Sequence[torch.Tensor], act: torch.Tensor
    ) -> list[torch.Tensor]: ...
    def targets(
        self,
        batch: Batch,
        main_net: nn.Module,
        target_nets: Sequence[nn.Module],
        cfg: TargetCfg,
    ) -> TargetOut: ...
    def loss(
        self,
        preds: Sequence[torch.Tensor],
        target: torch.Tensor,
        weights: torch.Tensor,
    ) -> LossOut: ...


class DQNAlgo:
    def infer_scores(self, net_out: torch.Tensor) -> torch.Tensor:
        return action_scores(net_out)

    def pred_for_actions(
        self, net_outs: Sequence[torch.Tensor], act: torch.Tensor
    ) -> list[torch.Tensor]:
        out = []
        for net_out in net_outs:
            q = net_out.gather(1, act.unsqueeze(1)).squeeze(1)
            out.append(q)
        return out

    def targets(
        self,
        batch: Batch,
        main_net: nn.Module,
        target_nets: Sequence[nn.Module],
        cfg: TargetCfg,
    ) -> TargetOut:
        return build_scalar_targets(batch, main_net, target_nets, cfg)

    def loss(
        self,
        preds: Sequence[torch.Tensor],
        target: torch.Tensor,
        weights: torch.Tensor,
    ) -> LossOut:
        loss, td_abs = dqn_loss(preds, target, weights)
        return LossOut(loss=loss, td_abs=td_abs, extras={})


class QRDQNAlgo:
    def __init__(self, n_quantiles: int = 32, device: torch.device | None = None):
        self._device = device or torch.device("cpu")
        self._tau = make_tau_hat(n_quantiles, self._device)

    def infer_scores(self, net_out: torch.Tensor) -> torch.Tensor:
        return action_scores(net_out)

    def pred_for_actions(
        self, net_outs: Sequence[torch.Tensor], act: torch.Tensor
    ) -> list[torch.Tensor]:
        assert len(net_outs) == 1
        net_out = net_outs[0]
        N = net_out.size(-1)
        pred = net_out.gather(
            1, act.unsqueeze(1).unsqueeze(2).expand(-1, 1, N)
        ).squeeze(1)
        return [pred]

    def targets(
        self,
        batch: Batch,
        main_net: nn.Module,
        target_nets: Sequence[nn.Module],
        cfg: TargetCfg,
    ) -> TargetOut:
        assert len(target_nets) == 1
        return build_quantile_targets(batch, main_net, target_nets[0], cfg)

    def loss(
        self,
        preds: Sequence[torch.Tensor],
        target: torch.Tensor,
        weights: torch.Tensor,
    ) -> LossOut:
        assert len(preds) == 1
        loss, td_abs = qrdqn_loss(
            preds[0], target, self._tau, weights, kappa=1.0
        )
        return LossOut(loss=loss, td_abs=td_abs, extras={})


def make_dqn_algo(
    use_distributional: bool = False,
    use_twin_q: bool = False,
    n_quantiles: int = 32,
    device: torch.device | None = None,
) -> Algo:
    if use_distributional:
        return QRDQNAlgo(n_quantiles=n_quantiles, device=device)
    return DQNAlgo()
