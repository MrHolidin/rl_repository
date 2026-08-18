"""Lobby round flow: turn handoff and battle resolution (2-player v0)."""

from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

import numpy as np

from src.bg_catalog.patch_context import PatchContext
from src.bg_catalog.ruleset import Ruleset
from src.bg_combat.battle import simulate_battle
from src.bg_core.board_helpers import snapshot_warband
from src.bg_core.minion import Race
from src.bg_lobby.player import PlayerPhase, PlayerState, apply_hero_damage
from src.bg_lobby.shop_order import sample_shop_turn_order
from src.bg_recruitment.combat_seat import PlayerCombatSeat
from src.bg_recruitment.economy import accrue_upgrade_discount, start_of_turn_gold
from src.bg_recruitment.hand_slots import apply_combat_hand_adds

if TYPE_CHECKING:
    from src.envs.minibg.state import MiniBGState


def after_player_finished(
    state: MiniBGState,
    idx: int,
    *,
    fire_on_turn_end: Callable[[PlayerState], None],
    resolve_battle_and_advance: Callable[[MiniBGState], None],
) -> None:
    """Called after a player transitions to DONE."""
    fire_on_turn_end(state.players[idx])
    order = state.shop_turn_order
    try:
        pos = order.index(idx)
    except ValueError as e:
        raise ValueError(f"player {idx} not in shop_turn_order {order}") from e
    for j in range(pos + 1, len(order)):
        next_idx = order[j]
        if state.players[next_idx].phase != PlayerPhase.DONE:
            state.current_player_index = next_idx
            return
    resolve_battle_and_advance(state)


def resolve_battle_and_advance(
    state: MiniBGState,
    *,
    rng: np.random.Generator,
    combat_board_max: int,
    board_size: int,
    ruleset: Ruleset,
    fire_on_turn_start: Callable[[PlayerState], None],
    refresh_shop: Callable[[PlayerState, Optional[Race]], None],
    refresh_shop_fill_empty_slots: Callable[[PlayerState, Optional[Race]], None],
    patch: Optional[PatchContext] = None,
) -> None:
    p0_has_initiative = (state.round_number % 2 == 1) == (state.initiative_player == 0)
    if patch is None:
        if state.shared_pool is None:
            raise RuntimeError(
                "patch or shared_pool is required for battle resolution"
            )
        patch = state.shared_pool.patch
    pa, pb = state.players[0], state.players[1]
    pa.last_opponent_board = snapshot_warband(pb.board)
    pb.last_opponent_board = snapshot_warband(pa.board)
    combat_gold = [0, 0]
    combat_hand_adds: list[list[str]] = [[], []]
    # minibg is a 1v1 sandbox — "alive count" for the top-N damage-cap lift
    # doesn't carry the 8-player "down to top 4" meaning, but is always 2
    # here, so a ruleset that enables the lift at N>=2 applies it from the
    # start (matches the 8-player lobby's field-of-2 endgame framing).
    damage_cap = ruleset.effective_damage_cap(state.round_number, 2)
    dmg_p0, dmg_p1 = simulate_battle(
        pa.board,
        pb.board,
        p0_has_initiative=p0_has_initiative,
        rng=rng,
        combat_board_max=combat_board_max,
        damage_cap=damage_cap,
        max_board_slots=board_size,
        p0_tavern_tier=state.players[0].tavern_tier,
        p1_tavern_tier=state.players[1].tavern_tier,
        patch=patch,
        combat_gold_out=combat_gold,
        combat_hand_adds_out=combat_hand_adds,
        # Seats let a fight write the things a combat copy cannot carry —
        # permanent Blood Gems, "this game" modifiers — straight onto the two
        # players. Gold and hand adds keep coming back through the lists above,
        # because those are applied in one batch after the fight by rule.
        seats=(PlayerCombatSeat(pa, patch=patch), PlayerCombatSeat(pb, patch=patch)),
    )
    apply_hero_damage(state.players[0], dmg_p0, patch=patch)
    apply_hero_damage(state.players[1], dmg_p1, patch=patch)
    state.players[0].gold += combat_gold[0]
    state.players[1].gold += combat_gold[1]
    apply_combat_hand_adds(pa, combat_hand_adds[0], patch, shared_pool=state.shared_pool)
    apply_combat_hand_adds(pb, combat_hand_adds[1], patch, shared_pool=state.shared_pool)
    state.players[0].last_combat_won = dmg_p0 == 0 and dmg_p1 > 0
    state.players[1].last_combat_won = dmg_p1 == 0 and dmg_p0 > 0

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

    if state.round_number >= ruleset.max_rounds:
        state.done = True
        state.winner = 0
        return

    state.round_number += 1
    for p in state.players:
        accrue_upgrade_discount(p)
        p.gold = start_of_turn_gold(p, state.round_number)
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
        # A freeze lasts until the shop it protected is served, then lifts: the
        # kept minions stay on the counter but are no longer pinned, so the next
        # roll clears them. Only the whole-shop flag above used to be cleared
        # here; the per-slot tuple was never cleared anywhere, and rolling with
        # it set preserved those slots for the rest of the game. That went
        # unnoticed because the per-action state copy dropped the field.
        p.shop_frozen = (False,) * len(p.shop_frozen)

    order = sample_shop_turn_order(rng, 2)
    state.shop_turn_order = (order[0], order[1])
    state.current_player_index = order[0]


__all__ = ["after_player_finished", "resolve_battle_and_advance"]
