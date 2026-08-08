"""Routing of flat agent metrics into per-group CSV files.

Replaces the old single wide ``metrics.csv`` (one row, 60-80 columns, most blank)
with several narrow files grouped by concern: ``metrics_core.csv``,
``metrics_critic.csv``, ``metrics_dvd.csv``, etc. Each file is created lazily —
only when its group actually produces data — so a non-DvD run never grows empty
``dvd_*`` columns.

A metric key is routed to a group, in order:
  1. a ``group/metric`` slash prefix -> a long-format file named after the prefix
     (future namespaced metrics, e.g. ``rnd/novelty_mean`` -> ``metrics_rnd.csv``);
  2. the static :data:`GROUP_OF` registry (the known wide groups below);
  3. the DvD per-identity array pattern -> the long-format ``dvd_identities`` file;
  4. otherwise the catch-all long-format ``misc`` file (never silently dropped).

To add a metric to a wide group: add its name to that group's tuple in
:data:`GROUP_COLUMNS`. Unregistered metrics still land in ``misc`` rather than
vanishing, so forgetting this step is visible, not silent.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

# Leading join columns written to every group file.
INDEX_FIELDS: Tuple[str, ...] = ("step", "episode")

# Wide groups: name -> ordered metric columns (declaration order = column order).
# Each becomes ``metrics_<name>.csv`` with a fixed header (no rewrites).
GROUP_COLUMNS: Dict[str, Tuple[str, ...]] = {
    # Optimisation scalars shared by PPO and DQN.
    "core": (
        "learning_rate",
        "epsilon",
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
        # Ordering-head entropy, logged apart from the action-head `entropy` it
        # is added to: the two have very different ceilings (ln(10.9)=2.39 vs
        # ln(7!)=8.53), so a single summed column would hide which head is
        # actually collapsing.
        "entropy_order",
        "entropy_coef",
        "order_entropy_coef",
        "approx_kl",
        "clip_frac",
        "grad_norm",
    ),
    # Critic quality / advantage statistics.
    "critic": (
        "return_mean",
        "advantage_mean",
        "advantage_std",
        "explained_variance",
        "placement_acc",
    ),
    # Rollout / replay bookkeeping.
    "rollout": (
        "rollout_size",
        "rollout_capacity",
        "buffer_utilization",
        "buffer_size",
        "buffer_capacity",
    ),
    # v4+ recurrent PPO (BPTT) diagnostics.
    "bptt": (
        "bptt_sequences",
        "bptt_seqs_per_mb",
        "bptt_mean_seq_len",
    ),
    # Auxiliary battle-prediction head.
    "battle_pred": (
        "battle_pred_loss",
        "battle_pred_mae",
        "battle_pred_corr",
        "battle_pred_sign_acc",
    ),
    # Auxiliary relative-strength head. One MAE column per configured horizon;
    # the names are built from the horizons, so list the defaults (1, 2, 4) and
    # let an unlisted horizon fall through to the catch-all rather than being
    # dropped silently.
    "strength_pred": (
        "strength_pred_loss",
        "strength_mae_h1",
        "strength_mae_h2",
        "strength_mae_h4",
    ),
    # DvD population-diversity aggregates (per-identity arrays go to dvd_identities).
    "dvd": (
        "dvd_pop_diversity",
        "dvd_identity_coverage",
        "dvd_distinct_tribes",
        "dvd_placement_best",
        "dvd_placement_worst",
        "dvd_placement_spread",
        "dvd_mean_assigned_frac",
        "dvd_mean_bonus",
        "dvd_bonus_place_ratio",
        "dvd_identity_contrib_norm",
    ),
    # DQN value-head statistics.
    "dqn": (
        "avg_q",
        "avg_target_q",
        "target_q_p95",
        "target_q_max",
        "td_error",
        "td_error_p95",
        "td_error_max",
        "q_spread",
        "top2_gap",
        "update_magnitude",
        "effective_step_ratio",
        "effective_step_size",
    ),
}

# DvD per-identity arrays (``dvd_place_3``, ``dvd_tribe_3``, ``dvd_assigned_frac_3``)
# are pivoted to a long file: one row per (step, identity) instead of 3*N columns.
DVD_IDENTITIES_GROUP = "dvd_identities"
DVD_IDENTITY_FIELDS: Tuple[str, ...] = ("place", "tribe", "assigned_frac")
_DVD_IDENTITY_RE = re.compile(r"^dvd_(place|tribe|assigned_frac)_(\d+)$")

# Catch-all long-format file for anything unregistered (step, episode, metric, value).
MISC_GROUP = "misc"

# Control signals carried in the metrics dict that are not numeric metrics.
IGNORED_KEYS = frozenset(
    {
        "checkpoint_saved",
        "target_network_updated",
        "battle_pred_config",
    }
)

# Reverse map: metric name -> wide group.
GROUP_OF: Dict[str, str] = {
    name: group for group, names in GROUP_COLUMNS.items() for name in names
}

WIDE_GROUPS: Tuple[str, ...] = tuple(GROUP_COLUMNS.keys())


def parse_dvd_identity(name: str) -> Optional[Tuple[str, int]]:
    """``(field, identity_index)`` for a DvD per-identity key, else ``None``."""
    m = _DVD_IDENTITY_RE.match(name)
    if m is None:
        return None
    return m.group(1), int(m.group(2))


def route(name: str) -> str:
    """Group name for a metric key (see module docstring)."""
    if "/" in name:
        return name.split("/", 1)[0]
    if name in GROUP_OF:
        return GROUP_OF[name]
    if parse_dvd_identity(name) is not None:
        return DVD_IDENTITIES_GROUP
    return MISC_GROUP


def display_name(name: str) -> str:
    """Column/label for a metric within its long-format group (strip slash prefix)."""
    return name.split("/", 1)[1] if "/" in name else name


def is_wide_group(group: str) -> bool:
    return group in GROUP_COLUMNS


def header_for(group: str) -> Tuple[str, ...]:
    """CSV header for a group file."""
    if group in GROUP_COLUMNS:
        return INDEX_FIELDS + GROUP_COLUMNS[group]
    if group == DVD_IDENTITIES_GROUP:
        return INDEX_FIELDS + ("identity",) + DVD_IDENTITY_FIELDS
    return INDEX_FIELDS + ("metric", "value")


__all__ = [
    "INDEX_FIELDS",
    "GROUP_COLUMNS",
    "GROUP_OF",
    "WIDE_GROUPS",
    "DVD_IDENTITIES_GROUP",
    "DVD_IDENTITY_FIELDS",
    "MISC_GROUP",
    "IGNORED_KEYS",
    "parse_dvd_identity",
    "route",
    "display_name",
    "is_wide_group",
    "header_for",
]
