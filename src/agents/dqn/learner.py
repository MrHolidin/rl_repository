from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from ...utils.batch import Batch
from .algo import Algo, LossOut
from .targets import TargetCfg, TargetOut, OptimizeCfg


class DqnLearner:
    def __init__(
        self,
        main_nets: Sequence[nn.Module],
        target_nets: Sequence[nn.Module],
        optimizer: torch.optim.Optimizer,
        algo: Algo,
        target_cfg: TargetCfg,
        optimize_cfg: OptimizeCfg,
    ):
        self.main_nets = list(main_nets)
        self.target_nets = list(target_nets)
        self.optimizer = optimizer
        self.algo = algo
        self.target_cfg = target_cfg
        self.optimize_cfg = optimize_cfg

    def train_on_batch(
        self, batch: Batch, detailed_metrics: bool = True
    ) -> tuple[dict, torch.Tensor]:
        out_list = [
            net(batch.obs, legal_mask=batch.legal) for net in self.main_nets
        ]
        preds = self.algo.pred_for_actions(out_list, batch.act)
        tgt_out = self.algo.targets(
            batch, self.main_nets[0], self.target_nets, self.target_cfg
        )
        loss_out = self.algo.loss(preds, tgt_out.target, batch.weights)
        extras = {**tgt_out.extras, **loss_out.extras}

        loss = loss_out.loss
        td_errors_abs = loss_out.td_abs

        if self.optimize_cfg.value_reg_weight > 0:
            value_reg = self.optimize_cfg.value_reg_weight * sum(
                (p ** 2).mean() for p in preds
            ) / len(preds)
            loss = loss + value_reg

        self.optimizer.zero_grad()
        loss.backward()
        params = [p for net in self.main_nets for p in net.parameters()]
        total_grad_norm = torch.nn.utils.clip_grad_norm_(
            params, max_norm=self.optimize_cfg.grad_clip_norm
        )
        self.optimizer.step()

        lr = self.optimizer.param_groups[0].get("lr", 0.0)
        metrics: dict = {}
        metrics["loss"] = loss.item()
        metrics.update(extras)

        if detailed_metrics:
            pre_clip_norm = float(total_grad_norm)
            metrics["grad_norm"] = pre_clip_norm
            step_norm = min(pre_clip_norm, self.optimize_cfg.grad_clip_norm)
            update_magnitude = lr * step_norm
            metrics["update_magnitude"] = update_magnitude
            metrics["effective_step_ratio"] = (
                step_norm / pre_clip_norm if pre_clip_norm > 0 else 1.0
            )
            with torch.no_grad():
                param_norm = sum(
                    p.data.pow(2).sum().item() for p in params
                ) ** 0.5
            metrics["effective_step_size"] = (
                update_magnitude / param_norm if param_norm > 0 else 0.0
            )
            with torch.no_grad():
                target_q_value = tgt_out.target
                avg_q = (
                    preds[0].mean(dim=1).mean()
                    if preds[0].dim() == 2
                    else preds[0].mean()
                )
                avg_target_q = target_q_value.mean()
                mean_td_error = td_errors_abs.mean()
                target_q_np = target_q_value.cpu().float().numpy()
                tq_sorted = np.sort(target_q_np)
                target_q_p95 = float(np.percentile(tq_sorted, 95))
                target_q_max = float(target_q_value.max().item())
                td_np = td_errors_abs.cpu().float().numpy()
                td_sorted = np.sort(td_np)
                td_error_p95 = float(np.percentile(td_sorted, 95))
                td_error_max = float(td_errors_abs.max().item())
                q_for_metrics = self.algo.infer_scores(out_list[0])
                masked_max = q_for_metrics.masked_fill(
                    ~batch.legal, float("-inf")
                ).max(dim=1)[0]
                masked_min = q_for_metrics.masked_fill(
                    ~batch.legal, float("inf")
                ).min(dim=1)[0]
                spread = masked_max - masked_min
                spread = torch.where(
                    torch.isfinite(spread), spread, torch.zeros_like(spread)
                )
                q_spread = float(spread.mean().item())
                masked_q = q_for_metrics.masked_fill(~batch.legal, float("-inf"))
                top2 = masked_q.topk(2, dim=1)
                gap = top2.values[:, 0] - top2.values[:, 1]
                gap = torch.where(
                    torch.isfinite(gap), gap, torch.zeros_like(gap)
                )
                top2_gap = float(gap.mean().item())
            metrics.update(
                {
                    "avg_q": avg_q.item(),
                    "avg_target_q": avg_target_q.item(),
                    "target_q_p95": target_q_p95,
                    "target_q_max": target_q_max,
                    "td_error": mean_td_error.item(),
                    "td_error_p95": td_error_p95,
                    "td_error_max": td_error_max,
                    "q_spread": q_spread,
                    "top2_gap": top2_gap,
                }
            )

        return metrics, td_errors_abs
