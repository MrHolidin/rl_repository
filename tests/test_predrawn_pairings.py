"""Pairings (incl. ghost pick) are pre-drawn at round start and observable
during the shop phase; the resolved combat fights exactly the pre-drawn pairs."""

from __future__ import annotations

import src.envs  # noqa: F401  (break circular import)
from src.agents.random_agent import RandomAgent
from src.bg_lobby.match_types import GHOST_OPPONENT_ID
from src.bg_lobby.pairing import opponent_from_pairings
from src.envs.bglike.lobby_env import BGLobbyEnv
from src.envs.bglike.seat_config import lobby_from_learned_seats

PATCH_DIR = "data/bgcore/19_6_0_74257"


def _make_env(seed: int) -> BGLobbyEnv:
    cfgs = lobby_from_learned_seats(
        (0,), agent_by_seat={0: RandomAgent(seed=seed)}, seed=seed
    )
    return BGLobbyEnv(
        cfgs,
        learned_seats=(0,),
        seed=seed,
        patch_dir=PATCH_DIR,
        obs_kind="bglike_v5",
    )


def test_initial_state_has_full_lobby_pairings():
    env = _make_env(11)
    s = env.reset()
    assert len(s.pairings) == 4
    assert all(not m.is_ghost for m in s.pairings)
    seats = {m.a for m in s.pairings} | {m.b for m in s.pairings}
    assert seats == set(range(8))
    for seat in range(8):
        assert opponent_from_pairings(s.pairings, seat) is not None


def test_predrawn_pairings_are_fought_and_ghost_known():
    env = _make_env(23)
    env.reset()
    saw_ghost_round = False
    checked_rounds = 0
    steps = 0
    while not env.lobby_done and steps < 30000:
        steps += 1
        s = env.state
        # Snapshot the pre-drawn pairings during the shop phase.
        pre_pairs = [(m.a, m.b) for m in s.pairings if m.b is not None]
        pre_ghosts = [m.a for m in s.pairings if m.is_ghost]
        combat_before = s.combat_round

        # Every alive seat must know its opponent during the shop phase.
        for seat in s.alive:
            opp = opponent_from_pairings(s.pairings, seat)
            assert opp is not None, (
                f"seat {seat} has unknown opponent at combat_round "
                f"{s.combat_round} (alive={s.alive})"
            )
        if pre_ghosts:
            saw_ghost_round = True

        cur = env.current_seat()
        if not env._seat_can_act(cur):
            break
        env.step_auto(cur)

        if env.state.combat_round > combat_before:
            # Combat resolved — it must have fought exactly the pre-drawn pairs.
            s2 = env.state
            for a, b in pre_pairs:
                assert s2.recent_opponents[a][-1] == b
                assert s2.recent_opponents[b][-1] == a
            for g in pre_ghosts:
                assert s2.recent_opponents[g][-1] == GHOST_OPPONENT_ID
            checked_rounds += 1

    assert checked_rounds >= 5
    assert saw_ghost_round, "no odd-alive (ghost) round encountered; change seed"


def test_ghost_match_opponent_resolves_to_dead_seat():
    env = _make_env(37)
    env.reset()
    steps = 0
    while not env.lobby_done and steps < 30000:
        steps += 1
        s = env.state
        for m in s.pairings:
            if m.is_ghost:
                opp = opponent_from_pairings(s.pairings, m.a)
                assert opp == m.ghost.seat
                assert opp not in s.alive
                return
        cur = env.current_seat()
        if not env._seat_can_act(cur):
            break
        env.step_auto(cur)
    raise AssertionError("no ghost match found; change seed")
