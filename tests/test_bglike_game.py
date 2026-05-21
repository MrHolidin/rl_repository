"""BGLikeGame: 8 players, hand 10, mandatory shared pool."""

from __future__ import annotations

import numpy as np

from src.envs.bglike.actions import HAND_SIZE, NUM_PLAYERS, Action
from src.envs.bglike.game import BGLikeGame
from src.bg_lobby.player import PlayerPhase


def _finish_one_shop_round(game: BGLikeGame, state):
    """Every alive player ends shop once (``FINISH``), then combat resolves."""
    combat_before = state.combat_round
    while not state.done and state.combat_round == combat_before:
        idx = state.current_player_index
        p = state.players[idx]
        if p.phase != PlayerPhase.SHOP:
            break
        legal = game.legal_actions(state)
        assert int(Action.FINISH) in legal
        state = game.apply_action(state, int(Action.FINISH))
    return state


def test_initial_eight_players_hand_ten_shared_pool():
    game = BGLikeGame(seed=1)
    s = game.initial_state()
    assert len(s.players) == NUM_PLAYERS == 8
    assert len(s.alive) == 8
    assert s.shared_pool is not None
    assert len(s.recent_opponents) == 8
    for seat in range(NUM_PLAYERS):
        assert len(s.players[seat].hand) == HAND_SIZE
    assert set(s.shop_turn_order) == set(range(NUM_PLAYERS))
    assert len(s.shop_turn_order) == NUM_PLAYERS


def test_pool_reserved_for_initial_shops():
    game = BGLikeGame(seed=2)
    s = game.initial_state()
    pool = s.shared_pool
    assert pool is not None
    total = sum(pool.remaining.values())
    assert total < sum(pool._initial.values())


def test_full_shop_cycle_triggers_combat_pairings():
    game = BGLikeGame(seed=3)
    s = game.initial_state()
    assert s.round_number == 1
    assert s.combat_round == 0
    s = _finish_one_shop_round(game, s)
    assert s.combat_round == 1
    assert len(s.pairings) == 4
    assert all(not m.is_ghost for m in s.pairings)
    assert s.round_number == 2
    for seat in s.alive:
        assert s.players[seat].phase == PlayerPhase.SHOP


def test_second_combat_round_advances_rr_cycle():
    game = BGLikeGame(seed=4)
    s = game.initial_state()
    s = _finish_one_shop_round(game, s)
    first_pairs = {(m.a, m.b) for m in s.pairings if m.b is not None}
    s = _finish_one_shop_round(game, s)
    assert s.combat_round == 2
    second_pairs = {(m.a, m.b) for m in s.pairings if m.b is not None}
    assert first_pairs != second_pairs


def test_current_player_is_seat_index():
    game = BGLikeGame(seed=5)
    s = game.initial_state()
    assert game.current_player(s) == s.current_player_index
    assert 0 <= game.current_player(s) < NUM_PLAYERS
