"""Eight-player lobby: shop turn order, pairing, combat, elimination."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from src.bg_combat.battle import simulate_battle
from src.bg_core.board_helpers import snapshot_warband
from src.bg_core.minion import Minion, Race
from src.bg_lobby.match_types import CombatMatch, EliminatedSnapshot, GHOST_OPPONENT_ID
from src.bg_lobby.pairing import compute_pairings, record_combat_opponent
from src.bg_lobby.player import BattleSnapshot, PlayerPhase, PlayerState
from src.bg_lobby.shared_pool import SharedCardPool
from src.bg_recruitment import hero_passives
from src.bg_recruitment.hand_slots import apply_combat_hand_adds
from src.bg_recruitment.pool_ledger import on_eliminate_player
from src.envs.bglike.actions import gold_for_round


def _count_board_tribes(board: Sequence[Minion]) -> dict:
    """Per-race count over a board. ``Race.ALL`` minions count toward every
    specific tribe (mirrors real-BG tribe-lock semantics)."""
    from collections import Counter

    counts: Counter = Counter()
    all_count = 0
    for m in board:
        r = m.race
        if r is None:
            continue
        if r == Race.ALL:
            all_count += 1
            continue
        counts[r] += 1
    if all_count:
        for r in (
            Race.BEAST,
            Race.DEMON,
            Race.MECHANICAL,
            Race.MURLOC,
            Race.DRAGON,
            Race.PIRATE,
            Race.ELEMENTAL,
        ):
            counts[r] += all_count
    return dict(counts)


def _alive_indices(state) -> Tuple[int, ...]:
    return tuple(i for i in state.alive if state.players[i].health > 0)


def _sync_alive(state) -> None:
    """Drop dead seats from ``alive``; ensure eliminations have snapshots first."""
    for seat in list(state.alive):
        if state.players[seat].health <= 0 and not any(
            snap.seat == seat for snap in state.eliminated
        ):
            _eliminate_seat(state, seat, combat_round=max(1, state.combat_round))
    state.alive = _alive_indices(state)


def after_player_finished(
    state,
    idx: int,
    *,
    fire_on_turn_end: Callable[[PlayerState], None],
    resolve_combat_round: Callable[[object], None],
) -> None:
    fire_on_turn_end(state.players[idx])
    order = state.shop_turn_order
    try:
        pos = order.index(idx)
    except ValueError as e:
        raise ValueError(f"player {idx} not in shop_turn_order {order}") from e
    for j in range(pos + 1, len(order)):
        next_idx = order[j]
        if next_idx not in state.alive:
            continue
        if state.players[next_idx].phase != PlayerPhase.DONE:
            state.current_player_index = next_idx
            return
    for seat in order:
        if seat in state.alive and state.players[seat].phase != PlayerPhase.DONE:
            state.current_player_index = seat
            return
    resolve_combat_round(state)


def _initiative_for_pair(
    state,
    a: int,
    b: int,
) -> bool:
    """True if ``a`` attacks first in this pair."""
    if len(state.players[a].board) != len(state.players[b].board):
        return len(state.players[a].board) > len(state.players[b].board)
    return (state.round_number % 2 == 1) == (state.initiative_player == a)


def _initiative_for_ghost(state, live_seat: int, ghost: EliminatedSnapshot) -> bool:
    """True if the live player attacks first vs a ghost board."""
    live_len = len(state.players[live_seat].board)
    ghost_len = len(ghost.last_board)
    if live_len != ghost_len:
        return live_len > ghost_len
    return (state.round_number % 2 == 1) == (state.initiative_player == live_seat)


def _apply_hero_damage(
    state,
    seat: int,
    damage: int,
) -> None:
    if damage > 0:
        state.players[seat].health -= damage


def _eliminate_seat(
    state,
    seat: int,
    *,
    combat_round: int,
) -> None:
    if any(snap.seat == seat for snap in state.eliminated):
        alive = [i for i in state.alive if i != seat]
        state.alive = tuple(alive)
        return
    p = state.players[seat]
    if state.shared_pool is not None:
        on_eliminate_player(state.shared_pool, p)
    snap = EliminatedSnapshot(
        seat=seat,
        last_board=tuple(p.board),
        tavern_tier=p.tavern_tier,
        eliminated_combat_round=combat_round,
    )
    state.eliminated = state.eliminated + (snap,)
    alive = [i for i in state.alive if i != seat]
    state.alive = tuple(alive)


def draw_combat_pairings(state, *, rng: np.random.Generator) -> None:
    """Pre-draw pairings for the UPCOMING combat (incl. the ghost pick).

    Called at round start (and from ``initial_state``) so the shop phase
    observes the true matchup — the "next opponent" in the obs is always the
    seat actually fought, ghost rounds included.
    """
    matches, state.full_lobby_cycle_round = compute_pairings(
        state.alive,
        state.recent_opponents,
        state.eliminated,
        n_seats=len(state.players),
        full_lobby_cycle_round=state.full_lobby_cycle_round,
        rng=rng,
    )
    state.pairings = matches


def _pairings_valid(state) -> bool:
    """Pre-drawn pairings are stale if a participant died during the shop
    phase (hero self-damage, e.g. Wrath Weaver) or were never drawn."""
    if not state.pairings:
        return False
    alive = set(state.alive)
    for m in state.pairings:
        if m.a not in alive:
            return False
        if m.b is not None and m.b not in alive:
            return False
    return True


def resolve_combat_round(
    state,
    *,
    rng: np.random.Generator,
    combat_board_max: int,
    damage_cap: int,
    board_size: int,
    max_tier: int,
    level_up_discount_per_round: int,
    max_rounds: int,
    fire_on_turn_start: Callable[[PlayerState], None],
    refresh_shop: Callable[[PlayerState, Optional[Race]], None],
    refresh_shop_fill_empty_slots: Callable[[PlayerState, Optional[Race]], None],
) -> None:
    _sync_alive(state)
    if len(state.alive) <= 1:
        state.done = True
        state.winner = state.alive[0] if state.alive else None
        return

    # Snapshot tribe counts BEFORE combat mutates anything so the obs builder
    # can show the "≥4 of tribe" indicator for the round that just ended.
    # Eliminated players keep their stored snapshot (frozen at elimination time).
    for seat in state.alive:
        p = state.players[seat]
        p.last_round_tribe_counts = _count_board_tribes(p.board)

    # Pairings were pre-drawn at round start (visible to the shop phase). The
    # redraw fallback covers shop-phase deaths and manually built states.
    if not _pairings_valid(state):
        draw_combat_pairings(state, rng=rng)
    matches = state.pairings
    state.combat_round += 1

    if state.shared_pool is None:
        raise RuntimeError("shared_pool is required for combat resolution")
    patch = state.shared_pool.patch

    for match in matches:
        if match.is_ghost:
            assert match.b is None and match.ghost is not None
            live = state.players[match.a]
            ghost_board = list(match.ghost.last_board)
            p0_first = _initiative_for_ghost(state, match.a, match.ghost)
            battle_result = simulate_battle(
                live.board,
                ghost_board,
                p0_has_initiative=p0_first,
                rng=rng,
                combat_board_max=combat_board_max,
                damage_cap=damage_cap,
                max_board_slots=board_size,
                p0_tavern_tier=live.tavern_tier,
                p1_tavern_tier=match.ghost.tavern_tier,
                patch=patch,
                # Ghost boards carry no hero; only the live seat's hero applies.
                p0_attack_aura_all=hero_passives.hero_combat_attack_aura(live),
                p0_start_combat_keywords=hero_passives.hero_start_combat_keywords(live),
            )
            dmg_live = battle_result.damage_p0
            _apply_hero_damage(state, match.a, dmg_live)
            # Record battle-prediction-head snapshots for the live seat. Ghost
            # is treated like a normal opponent: own=side0 (live), opp=side1
            # (ghost board). raw_signed: +raw_p1 if I won (raw_damage_p1 is
            # the damage *I dealt*), -raw_p0 if I lost (raw_damage_p0 is dmg I took).
            initial = battle_result.snapshots[0]
            live.last_battle_snapshots = (
                BattleSnapshot(
                    own_board=initial.side0_board,
                    opp_board=initial.side1_board,
                    step_index=0,
                ),
            )
            live.last_attack_first = (battle_result.attack_first_side == 0)
            live.last_battle_raw_signed = float(
                battle_result.raw_damage_p1 - battle_result.raw_damage_p0
            )
            ghost_rec = record_combat_opponent(
                state.recent_opponents[match.a], GHOST_OPPONENT_ID
            )
            state.recent_opponents = tuple(
                ghost_rec if i == match.a else state.recent_opponents[i]
                for i in range(len(state.players))
            )
            continue

        assert match.b is not None
        a, b = match.a, match.b
        pa, pb = state.players[a], state.players[b]
        pa.last_opponent_board = snapshot_warband(pb.board)
        pb.last_opponent_board = snapshot_warband(pa.board)
        p0_first = _initiative_for_pair(state, a, b)
        combat_gold = [0, 0]
        combat_hand_adds: list[list[str]] = [[], []]
        battle_result = simulate_battle(
            pa.board,
            pb.board,
            p0_has_initiative=p0_first,
            rng=rng,
            combat_board_max=combat_board_max,
            damage_cap=damage_cap,
            max_board_slots=board_size,
            p0_tavern_tier=pa.tavern_tier,
            p1_tavern_tier=pb.tavern_tier,
            patch=patch,
            combat_gold_out=combat_gold,
            combat_hand_adds_out=combat_hand_adds,
            p0_attack_aura_all=hero_passives.hero_combat_attack_aura(pa),
            p1_attack_aura_all=hero_passives.hero_combat_attack_aura(pb),
            p0_start_combat_keywords=hero_passives.hero_start_combat_keywords(pa),
            p1_start_combat_keywords=hero_passives.hero_start_combat_keywords(pb),
        )
        dmg_a, dmg_b = battle_result.damage_p0, battle_result.damage_p1
        # Battle-prediction-head snapshots for both seats. Orientation is
        # per-seat: pa sees side0 as own, pb sees side1 as own. The signed raw
        # damage is "damage I dealt minus damage I took" with the uncapped
        # formula, so positive = won decisively, negative = lost decisively.
        _initial = battle_result.snapshots[0]
        pa.last_battle_snapshots = (
            BattleSnapshot(
                own_board=_initial.side0_board,
                opp_board=_initial.side1_board,
                step_index=0,
            ),
        )
        pb.last_battle_snapshots = (
            BattleSnapshot(
                own_board=_initial.side1_board,
                opp_board=_initial.side0_board,
                step_index=0,
            ),
        )
        pa.last_attack_first = (battle_result.attack_first_side == 0)
        pb.last_attack_first = (battle_result.attack_first_side == 1)
        pa.last_battle_raw_signed = float(
            battle_result.raw_damage_p1 - battle_result.raw_damage_p0
        )
        pb.last_battle_raw_signed = float(
            battle_result.raw_damage_p0 - battle_result.raw_damage_p1
        )
        if p0_first:
            _apply_hero_damage(state, a, dmg_a)
            _apply_hero_damage(state, b, dmg_b)
            pa.last_combat_won = dmg_a == 0 and dmg_b > 0
            pb.last_combat_won = dmg_b == 0 and dmg_a > 0
            pa.gold += combat_gold[0]
            pb.gold += combat_gold[1]
            apply_combat_hand_adds(pa, combat_hand_adds[0], patch)
            apply_combat_hand_adds(pb, combat_hand_adds[1], patch)
        else:
            _apply_hero_damage(state, b, dmg_b)
            _apply_hero_damage(state, a, dmg_a)
            pa.last_combat_won = dmg_a == 0 and dmg_b > 0
            pb.last_combat_won = dmg_b == 0 and dmg_a > 0
            pa.gold += combat_gold[0]
            pb.gold += combat_gold[1]
            apply_combat_hand_adds(pa, combat_hand_adds[0], patch)
            apply_combat_hand_adds(pb, combat_hand_adds[1], patch)

        new_recent: List[Tuple[int, ...]] = []
        for i in range(len(state.players)):
            if i == a:
                new_recent.append(record_combat_opponent(state.recent_opponents[a], b))
            elif i == b:
                new_recent.append(record_combat_opponent(state.recent_opponents[b], a))
            else:
                new_recent.append(state.recent_opponents[i])
        state.recent_opponents = tuple(new_recent)

    for seat in list(state.alive):
        if state.players[seat].health <= 0:
            _eliminate_seat(state, seat, combat_round=state.combat_round)

    _sync_alive(state)
    if len(state.alive) <= 1:
        state.done = True
        state.winner = state.alive[0] if state.alive else None
        return

    if state.round_number >= max_rounds:
        state.done = True
        state.winner = state.alive[0]
        return

    state.round_number += 1
    for seat in state.alive:
        p = state.players[seat]
        if p.tavern_tier < max_tier:
            p.next_tier_up_cost = max(
                0, p.next_tier_up_cost - level_up_discount_per_round
            )
        p.gold = gold_for_round(state.round_number)
        p.phase = PlayerPhase.SHOP
        p.shop_actions_used = 0
        p.pending_choice = None
        p.triple_reward_discover_pending = False
        p.triple_reward_spell_tier = 0
        p.placed_minion_board_index = None
        p.placed_minion_pending_after = None
        fire_on_turn_start(p)
        # Hero per-turn levers (Nozdormu free roll, Rat King tribe rotation,
        # A.F. Kay gold-0 rounds). No-op when the seat has no hero.
        hero_passives.apply_hero_on_turn_start(
            p,
            state.round_number,
            patch=patch,
            rng=rng,
            shop_excluded_race=state.shop_excluded_race,
        )
        if p.shop_freeze_next_round:
            refresh_shop_fill_empty_slots(p, state.shop_excluded_race)
            p.shop_freeze_next_round = False
        else:
            refresh_shop(p, state.shop_excluded_race)

    from src.bg_lobby.shop_order import sample_shop_turn_order

    alive_list = list(state.alive)
    perm = sample_shop_turn_order(rng, len(alive_list))
    state.shop_turn_order = tuple(alive_list[i] for i in perm)
    state.current_player_index = state.shop_turn_order[0]

    # Draw next combat's pairings now — the upcoming shop phase observes them.
    draw_combat_pairings(state, rng=rng)


__all__ = ["after_player_finished", "draw_combat_pairings", "resolve_combat_round"]
