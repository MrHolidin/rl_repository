"""Backward-compatible imports; RL uses ``envs.minibg.rl_effects``."""

from __future__ import annotations

from src.bg_recruitment.effect_modal import (
    adjacent_board_indices,
    caster_ref_from_board_minion,
    compute_eligible_buff_target,
)

eligible_board_indices = compute_eligible_buff_target

__all__ = [
    "adjacent_board_indices",
    "caster_ref_from_board_minion",
    "compute_eligible_buff_target",
    "eligible_board_indices",
]
