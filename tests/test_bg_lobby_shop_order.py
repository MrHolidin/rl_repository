"""Lobby shop_turn_order: random permutation and current_player alignment."""

import numpy as np

from src.bg_lobby.shop_order import sample_shop_turn_order
from src.envs.minibg.game import MiniBGGame
from src.envs.minibg.state import PlayerPhase


def test_sample_shop_turn_order_is_permutation_of_two():
    order = sample_shop_turn_order(np.random.default_rng(0), 2)
    assert order in ((0, 1), (1, 0))


def test_initial_state_current_matches_shop_turn_order():
    g = MiniBGGame(seed=11)
    s = g.initial_state()
    assert s.shop_turn_order in ((0, 1), (1, 0))
    assert s.current_player_index == s.shop_turn_order[0]


def test_after_advance_current_matches_shop_turn_order():
    orders: list[tuple[int, int]] = []
    currents: list[int] = []
    for _ in range(2):
        g = MiniBGGame(seed=7)
        s = g.initial_state()
        s.players[0].phase = PlayerPhase.DONE
        s.players[1].phase = PlayerPhase.DONE
        g._resolve_battle_and_advance(s)
        assert not s.done
        assert s.shop_turn_order in ((0, 1), (1, 0))
        assert s.current_player_index in (0, 1)
        assert s.current_player_index == s.shop_turn_order[0]
        orders.append(s.shop_turn_order)
        currents.append(s.current_player_index)
    assert orders[0] == orders[1]
    assert currents[0] == currents[1]
