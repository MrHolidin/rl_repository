"""Shared test helpers for MiniBG combat defaults."""

from __future__ import annotations

from typing import Any

from src.bg_combat.battle import simulate_battle as _simulate_battle
from src.envs.minibg.actions import BOARD_SIZE, COMBAT_BOARD_MAX, DAMAGE_CAP


def simulate_battle(
    board_p0: list,
    board_p1: list,
    *,
    p0_has_initiative: bool,
    rng: Any,
    **kwargs: Any,
) -> tuple[int, int]:
    opts = {
        "combat_board_max": COMBAT_BOARD_MAX,
        "damage_cap": DAMAGE_CAP,
        "max_board_slots": BOARD_SIZE,
    }
    opts.update(kwargs)
    return _simulate_battle(
        board_p0,
        board_p1,
        p0_has_initiative=p0_has_initiative,
        rng=rng,
        **opts,
    )
