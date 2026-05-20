"""Two-player round flow: turn handoff and battle resolution."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from src.bg_combat.battle import simulate_battle
from src.bg_core.minion import Race
from src.envs.minibg.actions import gold_for_round
from src.envs.minibg.state import MiniBGState, PlayerPhase, PlayerState


def after_player_finished(
    state: MiniBGState,
    idx: int,
    *,
    fire_on_turn_end: Callable[[PlayerState], None],
    resolve_battle_and_advance: Callable[[MiniBGState], None],
) -> None:
    """Called after a player transitions to DONE."""
    fire_on_turn_end(state.players[idx])
    other_idx = 1 - idx
    if state.players[other_idx].phase != PlayerPhase.DONE:
        state.current_player_index = other_idx
    else:
        resolve_battle_and_advance(state)


def resolve_battle_and_advance(
    state: MiniBGState,
    *,
    rng: np.random.Generator,
    combat_board_max: int,
    damage_cap: int,
    board_size: int,
    max_rounds: int,
    max_tier: int,
    level_up_discount_per_round: int,
    fire_on_turn_start: Callable[[PlayerState], None],
    refresh_shop: Callable[[PlayerState, Optional[Race]], None],
    refresh_shop_fill_empty_slots: Callable[[PlayerState, Optional[Race]], None],
) -> None:
    p0_has_initiative = (state.round_number % 2 == 1) == (state.initiative_player == 0)
    dmg_p0, dmg_p1 = simulate_battle(
        state.players[0].board,
        state.players[1].board,
        p0_has_initiative=p0_has_initiative,
        rng=rng,
        combat_board_max=combat_board_max,
        damage_cap=damage_cap,
        max_board_slots=board_size,
        p0_tavern_tier=state.players[0].tavern_tier,
        p1_tavern_tier=state.players[1].tavern_tier,
    )
    state.players[0].health -= dmg_p0
    state.players[1].health -= dmg_p1

    p0_dead = state.players[0].health <= 0
    p1_dead = state.players[1].health <= 0

    if p0_dead or p1_dead:
        state.done = True
        if p0_dead and p1_dead:
            state.winner = 0
        elif p0_dead:
            state.winner = -1
        else:
            state.winner = 1
        return

    if state.round_number >= max_rounds:
        state.done = True
        state.winner = 0
        return

    state.round_number += 1
    for p in state.players:
        if p.tavern_tier < max_tier:
            p.next_tier_up_cost = max(0, p.next_tier_up_cost - level_up_discount_per_round)
        p.gold = gold_for_round(state.round_number)
        p.phase = PlayerPhase.SHOP
        p.shop_actions_used = 0
        p.pending_choice = None
        p.triple_reward_discover_pending = False
        p.triple_reward_spell_tier = 0
        p.placed_minion_board_index = None
        p.placed_minion_pending_after = None
        fire_on_turn_start(p)
        if p.shop_freeze_next_round:
            refresh_shop_fill_empty_slots(p, state.shop_excluded_race)
            p.shop_freeze_next_round = False
        else:
            refresh_shop(p, state.shop_excluded_race)
    state.current_player_index = 0
