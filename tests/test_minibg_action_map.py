import pytest

from src.envs.minibg.action_map import (
    A_APPLY_EFFECT_SKIP,
    A_BUY_BASE,
    A_DISCOVER_BASE,
    A_FINISH,
    A_FINISH_FREEZE_SHOP,
    A_LEVEL_UP,
    A_MAGNET_BASE,
    A_PLACE_BASE,
    A_ROLL,
    A_SELL_BASE,
    A_SWAP_BOARD_0,
    NUM_ACTIONS,
    NUM_ENV_ACTIONS,
    NUM_SWAP_ADJ,
    env_action_to_game_action,
    is_apply_effect_skip,
)
from src.envs.minibg.actions import Action as GameAction
from src.envs.minibg.actions import BOARD_SIZE, HAND_SIZE, MAX_SHOP_SLOTS


def test_action_layout():
    assert A_BUY_BASE == int(GameAction.BUY_SLOT_0)
    assert A_ROLL == int(GameAction.ROLL)
    assert A_LEVEL_UP == int(GameAction.LEVEL_UP)
    assert A_FINISH == int(GameAction.FINISH)
    assert A_FINISH_FREEZE_SHOP == int(GameAction.FINISH_FREEZE_SHOP)
    assert A_SELL_BASE == int(GameAction.SELL_BOARD_0)
    assert A_PLACE_BASE == int(GameAction.PLACE_HAND_0)
    assert A_MAGNET_BASE == int(GameAction.MAGNET_HAND_0_BOARD_0)
    assert A_DISCOVER_BASE == int(GameAction.DISCOVER_PICK_0)
    assert A_SWAP_BOARD_0 == NUM_ACTIONS
    assert A_APPLY_EFFECT_SKIP == A_SWAP_BOARD_0 + NUM_SWAP_ADJ
    assert NUM_ENV_ACTIONS == A_APPLY_EFFECT_SKIP + 1


def test_env_to_game_mapping_covers_all_non_order_actions():
    assert env_action_to_game_action(A_ROLL) == int(GameAction.ROLL)
    assert env_action_to_game_action(A_LEVEL_UP) == int(GameAction.LEVEL_UP)
    assert env_action_to_game_action(A_FINISH) == int(GameAction.FINISH)
    assert env_action_to_game_action(A_FINISH_FREEZE_SHOP) == int(GameAction.FINISH_FREEZE_SHOP)
    for s in range(MAX_SHOP_SLOTS):
        assert env_action_to_game_action(A_BUY_BASE + s) == int(GameAction.BUY_SLOT_0) + s
    for p in range(BOARD_SIZE):
        assert env_action_to_game_action(A_SELL_BASE + p) == int(GameAction.SELL_BOARD_0) + p
    for h in range(HAND_SIZE):
        assert env_action_to_game_action(A_PLACE_BASE + h) == int(GameAction.PLACE_HAND_0) + h
    for h in range(HAND_SIZE):
        for b in range(BOARD_SIZE):
            assert (
                env_action_to_game_action(A_MAGNET_BASE + h * BOARD_SIZE + b)
                == int(GameAction.MAGNET_HAND_0_BOARD_0) + h * BOARD_SIZE + b
            )
    for i in range(3):
        assert env_action_to_game_action(A_DISCOVER_BASE + i) == int(GameAction.DISCOVER_PICK_0) + i
    with pytest.raises(ValueError):
        env_action_to_game_action(A_SWAP_BOARD_0)
    with pytest.raises(ValueError):
        env_action_to_game_action(A_APPLY_EFFECT_SKIP)
    assert is_apply_effect_skip(A_APPLY_EFFECT_SKIP)
