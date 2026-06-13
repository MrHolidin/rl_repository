"""Battlegrounds-style combat pairing (8p round-robin, cooldown, ghost)."""

from __future__ import annotations

from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

from .match_types import GHOST_OPPONENT_ID, CombatMatch, EliminatedSnapshot

COOLDOWN_COMBAT_ROUNDS = 3
DEFAULT_LOBBY_SIZE = 8


def build_round_robin_schedule(n: int = DEFAULT_LOBBY_SIZE) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    """Circle-method schedule: ``n-1`` combat rounds, ``n/2`` pairs each (``n`` even).

    Player ``0`` is fixed; players ``1..n-1`` rotate. Each unordered pair appears once
    per full cycle (patch 33.2 full-lobby behaviour).
    """
    if n < 2 or n % 2 != 0:
        raise ValueError(f"n must be an even integer >= 2, got {n}")
    fixed = 0
    circle: List[int] = list(range(1, n))
    rounds: List[Tuple[Tuple[int, int], ...]] = []
    for _ in range(n - 1):
        lineup = [fixed] + circle
        pairs: List[Tuple[int, int]] = []
        half = n // 2
        for i in range(half):
            a, b = lineup[i], lineup[n - 1 - i]
            pairs.append((a, b) if a < b else (b, a))
        rounds.append(tuple(pairs))
        circle = [circle[-1]] + circle[:-1]
    return tuple(rounds)


def _recent_forbidden(
    a: int,
    b: int,
    recent_opponents: Sequence[Sequence[int]],
    *,
    cooldown: int,
) -> bool:
    """True if pairing ``(a, b)`` violates the last ``cooldown`` combat opponents."""
    ra = recent_opponents[a][-cooldown:] if a < len(recent_opponents) else ()
    rb = recent_opponents[b][-cooldown:] if b < len(recent_opponents) else ()
    return b in ra or a in rb


def _pairing_allowed(
    a: int,
    b: int,
    recent_opponents: Sequence[Sequence[int]],
    relax_level: int,
) -> bool:
    cooldown = max(0, COOLDOWN_COMBAT_ROUNDS - relax_level)
    if cooldown == 0:
        return True
    return not _recent_forbidden(a, b, recent_opponents, cooldown=cooldown)


def _max_matching(
    alive: Sequence[int],
    edges: Set[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Maximum cardinality matching on a small graph (backtracking)."""
    alive_list = sorted(alive)
    if not alive_list:
        return []

    best: List[Tuple[int, int]] = []

    def edge_ok(i: int, j: int) -> bool:
        lo, hi = (i, j) if i < j else (j, i)
        return (lo, hi) in edges

    def solve(unmatched: List[int], cur: List[Tuple[int, int]]) -> None:
        nonlocal best
        if not unmatched:
            if len(cur) > len(best):
                best = list(cur)
            return
        if len(cur) + len(unmatched) // 2 <= len(best):
            return
        u = unmatched[0]
        rest = unmatched[1:]
        # leave u for ghost (odd case handled outside)
        solve(rest, cur)
        for v in rest:
            if edge_ok(u, v):
                lo, hi = (u, v) if u < v else (v, u)
                solve([x for x in rest if x != v], cur + [(lo, hi)])

    solve(alive_list, [])
    return best


def _build_edges(
    alive: Sequence[int],
    recent_opponents: Sequence[Sequence[int]],
    relax_level: int,
) -> Set[Tuple[int, int]]:
    edges: Set[Tuple[int, int]] = set()
    als = sorted(alive)
    for i in range(len(als)):
        for j in range(i + 1, len(als)):
            a, b = als[i], als[j]
            if _pairing_allowed(a, b, recent_opponents, relax_level):
                edges.add((a, b))
    return edges


def _flexible_pairing(
    alive: Sequence[int],
    recent_opponents: Sequence[Sequence[int]],
) -> List[Tuple[int, int]]:
    """Pair alive players with 3-round cooldown, relaxing if no large matching exists."""
    if len(alive) < 2:
        return []
    for relax in range(COOLDOWN_COMBAT_ROUNDS + 1):
        edges = _build_edges(alive, recent_opponents, relax)
        matching = _max_matching(alive, edges)
        target_pairs = len(alive) // 2
        if len(matching) >= target_pairs:
            return matching
    # fallback: arbitrary greedy on last relax
    edges = _build_edges(alive, recent_opponents, COOLDOWN_COMBAT_ROUNDS)
    if not edges:
        als = sorted(alive)
        return [(als[0], als[1])] if len(als) >= 2 else []
    matching = _max_matching(alive, edges)
    return matching


def _ghost_forbidden(recent: Sequence[int], cooldown: int = COOLDOWN_COMBAT_ROUNDS) -> bool:
    return GHOST_OPPONENT_ID in recent[-cooldown:]


def _flexible_pairing_with_ghost(
    alive: Sequence[int],
    recent_opponents: Sequence[Sequence[int]],
    rng: np.random.Generator,
) -> Tuple[List[Tuple[int, int]], int]:
    """Return ``(live_pairs, ghost_seat)`` when ``len(alive)`` is odd."""
    alive_list = sorted(alive)
    ghost_order = list(alive_list)
    rng.shuffle(ghost_order)

    for ghost_seat in ghost_order:
        sub = [s for s in alive_list if s != ghost_seat]
        for relax in range(COOLDOWN_COMBAT_ROUNDS + 1):
            edges = _build_edges(sub, recent_opponents, relax)
            matching = _max_matching(sub, edges)
            if len(matching) == len(sub) // 2:
                if not _ghost_forbidden(recent_opponents[ghost_seat]):
                    return matching, ghost_seat

    ghost_seat = ghost_order[0]
    sub = [s for s in alive_list if s != ghost_seat]
    matching = _flexible_pairing(sub, recent_opponents)
    return matching, ghost_seat


def _pick_ghost_snapshot(
    eliminated: Sequence[EliminatedSnapshot],
    rng: np.random.Generator,
) -> EliminatedSnapshot:
    if not eliminated:
        raise ValueError("ghost combat requires at least one eliminated snapshot")
    idx = int(rng.integers(0, len(eliminated)))
    return eliminated[idx]


def record_combat_opponent(
    recent: Sequence[int],
    opponent: int,
    *,
    max_len: int = COOLDOWN_COMBAT_ROUNDS,
) -> Tuple[int, ...]:
    """Append opponent id (player seat or ``GHOST_OPPONENT_ID``), keep last ``max_len``."""
    out = tuple(recent) + (opponent,)
    if len(out) > max_len:
        out = out[-max_len:]
    return out


def compute_pairings(
    alive: Sequence[int],
    recent_opponents: Sequence[Sequence[int]],
    eliminated: Sequence[EliminatedSnapshot],
    *,
    n_seats: int = DEFAULT_LOBBY_SIZE,
    full_lobby_cycle_round: int = 0,
    rng: np.random.Generator,
) -> Tuple[Tuple[CombatMatch, ...], int]:
    """Build combat pairings for this round.

    Returns ``(matches, next_full_lobby_cycle_round)``.

    * ``len(alive) == n_seats`` (full lobby): fixed round-robin schedule.
    * ``len(alive) == 2``: always each other (cooldown waived).
    * otherwise: flexible matching + ghost if odd count.
    """
    alive_tuple = tuple(sorted(alive))
    if len(alive_tuple) < 1:
        return (), full_lobby_cycle_round

    if len(alive_tuple) == 2:
        a, b = alive_tuple[0], alive_tuple[1]
        return (CombatMatch(a, b, None),), full_lobby_cycle_round

    if len(alive_tuple) == n_seats and all(
        s in alive_tuple for s in range(n_seats)
    ):
        schedule = build_round_robin_schedule(n_seats)
        r = full_lobby_cycle_round % (n_seats - 1)
        pairs = schedule[r]
        matches = tuple(CombatMatch(a, b, None) for a, b in pairs)
        next_cycle = (r + 1) % (n_seats - 1)
        return matches, next_cycle

    if len(alive_tuple) % 2 == 1:
        if not eliminated:
            bye_idx = int(rng.integers(0, len(alive_tuple)))
            bye_seat = alive_tuple[bye_idx]
            sub = tuple(s for s in alive_tuple if s != bye_seat)
            pairs = _flexible_pairing(sub, recent_opponents)
            matches = [CombatMatch(a, b, None) for a, b in pairs]
            return tuple(matches), full_lobby_cycle_round

        pairs, ghost_seat = _flexible_pairing_with_ghost(
            alive_tuple, recent_opponents, rng
        )
        matches: List[CombatMatch] = [
            CombatMatch(a, b, None) for a, b in pairs
        ]
        snapshot = _pick_ghost_snapshot(eliminated, rng)
        matches.append(CombatMatch(ghost_seat, None, snapshot))
        return tuple(matches), full_lobby_cycle_round

    pairs = _flexible_pairing(alive_tuple, recent_opponents)
    matches = [CombatMatch(a, b, None) for a, b in pairs]
    return tuple(matches), full_lobby_cycle_round


def opponent_from_pairings(
    pairings: Sequence[CombatMatch],
    seat: int,
) -> Optional[int]:
    """``seat``'s opponent in pre-drawn pairings (ghost → the snapshot's seat).

    Returns None when ``seat`` has no match (finished / not in pairings) or
    ``pairings`` is empty (state built without ``draw_combat_pairings``).
    """
    for m in pairings:
        if m.a == seat:
            if m.b is not None:
                return int(m.b)
            if m.ghost is not None:
                return int(m.ghost.seat)
            return None
        if m.b == seat:
            return int(m.a)
    return None


def peek_next_opponent(
    alive: Sequence[int],
    recent_opponents: Sequence[Sequence[int]],
    *,
    n_seats: int = DEFAULT_LOBBY_SIZE,
    full_lobby_cycle_round: int = 0,
    seat: int,
) -> Optional[int]:
    """Return ``seat``'s opponent for the upcoming combat, or ``None`` if it is
    not deterministically known (odd-alive bye / ghost pick branches in
    ``compute_pairings`` are rng-driven).

    Mirrors ``compute_pairings`` for the deterministic branches only:
    full-lobby round-robin, 2-alive (forced pair), and even-count flexible
    pairing (which does not consume rng). The obs builder uses this in the
    shop phase to expose "who I'm fighting next" without advancing state rng.
    """
    alive_tuple = tuple(sorted(alive))
    if seat not in alive_tuple:
        return None
    if len(alive_tuple) < 2:
        return None
    if len(alive_tuple) == 2:
        return alive_tuple[1] if alive_tuple[0] == seat else alive_tuple[0]
    if len(alive_tuple) == n_seats and all(
        s in alive_tuple for s in range(n_seats)
    ):
        schedule = build_round_robin_schedule(n_seats)
        r = full_lobby_cycle_round % (n_seats - 1)
        pairs = schedule[r]
        for a, b in pairs:
            if a == seat:
                return b
            if b == seat:
                return a
        return None
    if len(alive_tuple) % 2 == 1:
        # Odd-alive: bye / ghost choice is rng-driven → not deterministic.
        return None
    pairs = _flexible_pairing(alive_tuple, recent_opponents)
    for a, b in pairs:
        if a == seat:
            return b
        if b == seat:
            return a
    return None


def opponents_met_in_cycle(
    cycle_rounds_played: int,
    n_seats: int = DEFAULT_LOBBY_SIZE,
) -> int:
    """How many distinct opponents each player has faced after ``cycle_rounds_played`` RR rounds."""
    return min(cycle_rounds_played, n_seats - 1)


__all__ = [
    "COOLDOWN_COMBAT_ROUNDS",
    "DEFAULT_LOBBY_SIZE",
    "GHOST_OPPONENT_ID",
    "CombatMatch",
    "opponent_from_pairings",
    "peek_next_opponent",
    "EliminatedSnapshot",
    "build_round_robin_schedule",
    "compute_pairings",
    "opponents_met_in_cycle",
    "record_combat_opponent",
]
