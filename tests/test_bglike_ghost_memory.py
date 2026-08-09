"""A ghost fight must overwrite the seat's remembered opponent board.

Only the two-live-seats branch of ``resolve_combat_round`` used to write
``PlayerState.last_opponent_board``. After a ghost round a seat therefore kept
the board of whoever it fought *before* the ghost — the memory silently aged by
a round, and ``AddFromLastOpponentBoardEffect`` (which draws a minion from "the
last opponent's board") drew from the wrong warband.

This needs its own test: the field is not part of the golden-trace hash, and it
reaches play through a single card, so neither the trace fixture nor the rest of
the suite would notice the difference.
"""

from __future__ import annotations

import pytest

from src.agents.random_agent import RandomAgent
from src.bg_catalog.patch_context import load_patch_context
from src.bg_core.board_helpers import snapshot_warband
from src.envs.bglike.lobby_env import BGLobbyEnv
from src.envs.bglike.seat_config import lobby_from_learned_seats

PATCH_DIR = "data/bgcore/19_6_0_74257"


def _first_ghost_fight(seed: int, max_steps: int = 4000):
    """Drive a lobby until a ghost pairing resolves.

    Returns ``(seat, board_seen, board_remembered)`` or None if no ghost came up.
    """
    agents = {s: RandomAgent(seed=seed * 10 + s) for s in range(8)}
    env = BGLobbyEnv(
        lobby_from_learned_seats(tuple(range(8)), agent_by_seat=agents),
        learned_seats=tuple(range(8)),
        training_seats=(0,),
        seed=seed,
        patch_dir=PATCH_DIR,
    )
    env.reset(seed=seed)

    pending = None  # (seat, ghost board, combat_round it was drawn for)
    for _ in range(max_steps):
        if env.lobby_done:
            break
        state = env.state
        if pending is not None and state.combat_round > pending[2]:
            seat, seen, _ = pending
            return seat, seen, tuple(state.players[seat].last_opponent_board)
        if pending is None:
            ghost = next((m for m in (state.pairings or ()) if m.is_ghost), None)
            if ghost is not None and ghost.ghost is not None:
                pending = (int(ghost.a), tuple(ghost.ghost.last_board), state.combat_round)

        seat = env.current_seat()
        if not env._seat_can_act(seat):
            break
        env.step_action(seat, int(agents[seat].act(
            env.obs_for_seat(seat), legal_mask=env.legal_mask_for_seat(seat)
        )))
    return None


def test_ghost_fight_updates_the_remembered_board():
    found = None
    for seed in range(25):
        found = _first_ghost_fight(seed)
        if found is not None:
            break
    if found is None:
        pytest.skip("no ghost pairing resolved in 25 seeds")

    seat, seen, remembered = found
    assert [m.card_id for m in remembered] == [m.card_id for m in seen], (
        f"seat {seat} fought a ghost but remembers a different board"
    )


def test_remembered_board_is_a_copy_not_a_live_reference():
    """What a seat remembers must not change when that minion is later buffed.

    ``last_battle_snapshots`` holds the live Minion objects, so sourcing the
    memory from there would let a later shop buff rewrite history. It goes
    through ``snapshot_warband`` for exactly this reason.
    """
    patch = load_patch_context(PATCH_DIR)
    live = [patch.make_minion("EX1_103")]
    remembered = snapshot_warband(live)
    before = remembered[0].raw_attack

    live[0].bonus_attack += 5
    assert remembered[0].raw_attack == before
    assert live[0].raw_attack == before + 5
