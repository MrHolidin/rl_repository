import pytest

from src.envs.minibg.action_map import (
    A_BUY_BASE,
    A_FINISH,
    A_LEVEL_UP,
    A_PLACE_BASE,
    A_ROLL,
    A_SELECT_ORDER_BASE,
    A_SELL_BASE,
    NUM_ENV_ACTIONS,
    NUM_PERMS,
    PERMUTATIONS_4,
    env_action_to_game_action,
)
from src.envs.minibg.actions import Action as GameAction
from src.envs.minibg.actions import BOARD_SIZE, HAND_SIZE, SHOP_SIZE


def test_action_layout():
    assert (A_ROLL, A_LEVEL_UP, A_BUY_BASE, A_SELL_BASE, A_PLACE_BASE, A_FINISH,
            A_SELECT_ORDER_BASE) == (0, 1, 2, 5, 9, 12, 13)
    assert NUM_ENV_ACTIONS == A_SELECT_ORDER_BASE + NUM_PERMS == 37
    assert PERMUTATIONS_4[0] == (0, 1, 2, 3)
    assert len(set(PERMUTATIONS_4)) == NUM_PERMS == 24


def test_env_to_game_mapping_covers_all_non_order_actions():
    assert env_action_to_game_action(A_ROLL) == int(GameAction.ROLL)
    assert env_action_to_game_action(A_LEVEL_UP) == int(GameAction.LEVEL_UP)
    assert env_action_to_game_action(A_FINISH) == int(GameAction.FINISH)
    for s in range(SHOP_SIZE):
        assert env_action_to_game_action(A_BUY_BASE + s) == int(GameAction.BUY_SLOT_0) + s
    for p in range(BOARD_SIZE):
        assert env_action_to_game_action(A_SELL_BASE + p) == int(GameAction.SELL_BOARD_0) + p
    for h in range(HAND_SIZE):
        assert env_action_to_game_action(A_PLACE_BASE + h) == int(GameAction.PLACE_HAND_0) + h
    with pytest.raises(ValueError):
        env_action_to_game_action(A_SELECT_ORDER_BASE)
