"""CSV column presets for MetricsFileCallback (DQN vs PPO vs minimal)."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple


# Leading columns populated from trainer step / agent introspection each row.
METRICS_CSV_PREFIX: Tuple[str, ...] = ("step", "episode", "epsilon", "learning_rate")

# Deep Q-training scalars emitted by DQNAgent.update (via learner / detailed metrics).
METRICS_PRESET_DQN: Tuple[str, ...] = (
    "avg_q",
    "avg_target_q",
    "target_q_p95",
    "target_q_max",
    "td_error",
    "td_error_p95",
    "td_error_max",
    "q_spread",
    "top2_gap",
    "grad_norm",
    "update_magnitude",
)

# On-policy PPO scalars from PPOAgent.update.
METRICS_PRESET_PPO: Tuple[str, ...] = (
    "loss",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_frac",
    "grad_norm",
    "rollout_size",
    "rollout_capacity",
    "buffer_utilization",
    "return_mean",
    "advantage_mean",
    "advantage_std",
    # Scheduled-hyperparam callbacks emit these when active; blank otherwise.
    "entropy_coef",
    # v4+ recurrent PPO emits these; blank for non-recurrent runs.
    "bptt_sequences",
    "bptt_seqs_per_mb",
    "bptt_mean_seq_len",
    # Auxiliary battle-prediction head; emitted only when ``battle_pred.enabled``.
    "battle_pred_loss",
    "battle_pred_mae",
    "battle_pred_corr",
    "battle_pred_sign_acc",
)

# Unknown agents: small generic set (both DQN and PPO often expose loss / grad_norm).
METRICS_PRESET_MINIMAL: Tuple[str, ...] = ("loss", "grad_norm")


def _dedupe_preserve_order(parts: Sequence[str]) -> Tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for x in parts:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return tuple(out)


def resolve_metrics_csv_fieldnames(
    agent_id: str,
    *,
    preset: str = "auto",
    columns: Optional[Sequence[str]] = None,
) -> Tuple[str, ...]:
    """
    Full ordered CSV header for ``MetricsFileCallback``.

    - If ``columns`` is a non-empty sequence, it is used as-is (full control).
    - Otherwise ``preset`` selects algorithm-specific columns after
      :data:`METRICS_CSV_PREFIX`.
    - ``preset=auto`` maps ``agent_id`` to ``dqn`` / ``ppo`` / ``minimal``.
    """
    if columns:
        cols = tuple(str(c).strip() for c in columns if str(c).strip())
        if not cols:
            raise ValueError("metrics_file columns, if set, must be a non-empty list")
        return cols

    p = (preset or "auto").strip().lower()
    if p == "auto":
        aid = (agent_id or "").strip().lower()
        if aid == "ppo":
            p = "ppo"
        elif aid == "dqn":
            p = "dqn"
        else:
            p = "minimal"

    if p == "dqn":
        suffix = METRICS_PRESET_DQN
    elif p == "ppo":
        suffix = METRICS_PRESET_PPO
    elif p == "minimal":
        suffix = METRICS_PRESET_MINIMAL
    else:
        raise ValueError(
            f"Unknown metrics_file preset {preset!r}. "
            f"Use 'auto', 'dqn', 'ppo', 'minimal', or pass explicit 'columns'."
        )

    return _dedupe_preserve_order(METRICS_CSV_PREFIX + suffix)


# Default header when ``MetricsFileCallback`` is constructed without ``fieldnames``
# (backward compatible with older call sites).
LEGACY_DQN_METRICS_FIELDS: Tuple[str, ...] = _dedupe_preserve_order(
    METRICS_CSV_PREFIX + METRICS_PRESET_DQN
)

__all__ = [
    "METRICS_CSV_PREFIX",
    "METRICS_PRESET_DQN",
    "METRICS_PRESET_PPO",
    "METRICS_PRESET_MINIMAL",
    "LEGACY_DQN_METRICS_FIELDS",
    "resolve_metrics_csv_fieldnames",
]
