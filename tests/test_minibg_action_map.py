import pytest

from src.envs.minibg.action_map import (
    A_BUY_BASE,
    A_LEVEL_UP,
    A_ROLL,
    A_SELECT_ORDER_BASE,
    A_SELL_BASE,
    NUM_ENV_ACTIONS,
    NUM_PERMS,
    PERMUTATIONS_4,
    buy_slot,
    env_action_to_game_action,
    is_buy,
    is_select_order,
    is_sell,
    legal_order_indices,
    order_index,
    sell_pos,
)
from src.envs.minibg.actions import Action as GameAction
from src.envs.minibg.actions import BOARD_SIZE


def test_action_indices_layout():
    assert NUM_ENV_ACTIONS == 33
    assert A_ROLL == 0
    assert A_LEVEL_UP == 1
    assert A_BUY_BASE == 2
    assert A_SELL_BASE == 5
    assert A_SELECT_ORDER_BASE == 9
    assert NUM_PERMS == 24


def test_permutations_are_all_distinct_and_complete():
    assert len(PERMUTATIONS_4) == NUM_PERMS
    assert len(set(PERMUTATIONS_4)) == NUM_PERMS
    for perm in PERMUTATIONS_4:
        assert sorted(perm) == list(range(BOARD_SIZE))


def test_identity_permutation_at_index_zero():
    assert PERMUTATIONS_4[0] == (0, 1, 2, 3)


def test_predicates_partition_action_space():
    counts = {"roll": 0, "level": 0, "buy": 0, "sell": 0, "order": 0}
    for a in range(NUM_ENV_ACTIONS):
        if a == A_ROLL:
            counts["roll"] += 1
        elif a == A_LEVEL_UP:
            counts["level"] += 1
        elif is_buy(a):
            counts["buy"] += 1
        elif is_sell(a):
            counts["sell"] += 1
        elif is_select_order(a):
            counts["order"] += 1
        else:
            pytest.fail(f"action {a} not classified")
    assert counts == {"roll": 1, "level": 1, "buy": 3, "sell": 4, "order": 24}


def test_buy_slot_and_sell_pos_indices():
    assert [buy_slot(a) for a in range(A_BUY_BASE, A_BUY_BASE + 3)] == [0, 1, 2]
    assert [sell_pos(a) for a in range(A_SELL_BASE, A_SELL_BASE + 4)] == [0, 1, 2, 3]


def test_order_index_range():
    for a in range(A_SELECT_ORDER_BASE, NUM_ENV_ACTIONS):
        j = order_index(a)
        assert 0 <= j < NUM_PERMS


def test_env_to_game_mapping_for_non_order_actions():
    assert env_action_to_game_action(A_ROLL) == int(GameAction.ROLL)
    assert env_action_to_game_action(A_LEVEL_UP) == int(GameAction.LEVEL_UP)
    for s in range(3):
        assert env_action_to_game_action(A_BUY_BASE + s) == int(GameAction.BUY_SLOT_0) + s
    for p in range(BOARD_SIZE):
        assert env_action_to_game_action(A_SELL_BASE + p) == int(GameAction.SELL_BOARD_0) + p


def test_env_to_game_rejects_order_actions():
    with pytest.raises(ValueError):
        env_action_to_game_action(A_SELECT_ORDER_BASE)
    with pytest.raises(ValueError):
        env_action_to_game_action(NUM_ENV_ACTIONS - 1)


def test_legal_order_indices_for_empty_board_only_identity():
    indices = legal_order_indices(0)
    assert indices == [0]
    assert PERMUTATIONS_4[indices[0]] == (0, 1, 2, 3)


def test_legal_order_indices_for_size_1_only_identity():
    indices = legal_order_indices(1)
    assert indices == [0]


def test_legal_order_indices_for_size_2_has_two():
    indices = legal_order_indices(2)
    perms = [PERMUTATIONS_4[i] for i in indices]
    assert len(perms) == 2
    assert (0, 1, 2, 3) in perms
    assert (1, 0, 2, 3) in perms
    for p in perms:
        assert p[2] == 2 and p[3] == 3


def test_legal_order_indices_for_size_3_has_six():
    indices = legal_order_indices(3)
    assert len(indices) == 6
    for i in indices:
        p = PERMUTATIONS_4[i]
        assert p[3] == 3
        assert sorted(p[:3]) == [0, 1, 2]


def test_legal_order_indices_for_full_board_has_all_24():
    indices = legal_order_indices(BOARD_SIZE)
    assert indices == list(range(NUM_PERMS))
