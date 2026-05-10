import numpy as np
import pytest

from src.envs.minibg.action_map import (
    A_BUY_BASE,
    A_LEVEL_UP,
    A_ROLL,
    A_SELECT_ORDER_BASE,
    A_SELL_BASE,
    NUM_ENV_ACTIONS,
    PERMUTATIONS_4,
    legal_order_indices,
)
from src.envs.minibg.actions import BOARD_SIZE, MAX_SHOP_ACTIONS
from src.envs.minibg.cards import make_minion
from src.envs.minibg.env import INVALID_ACTION_REWARD, MiniBGEnv
from src.envs.minibg.obs import OBS_DIM


def _set_shop(env: MiniBGEnv, idx: int, *card_ids):
    p = env._state.players[idx]
    p.shop = [make_minion(cid) if cid is not None else None for cid in card_ids]
    while len(p.shop) < 3:
        p.shop.append(None)


def test_reset_returns_obs_of_correct_shape():
    env = MiniBGEnv(seed=0)
    obs = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32


def test_legal_actions_mask_has_full_size_and_finish_options():
    env = MiniBGEnv(seed=0)
    mask = env.legal_actions_mask
    assert mask.shape == (NUM_ENV_ACTIONS,)
    assert mask.dtype == bool
    assert mask[A_ROLL]  # gold=3 ≥ ROLL_COST
    assert not mask[A_LEVEL_UP]  # gold=3 < 4
    assert mask[A_BUY_BASE] and mask[A_BUY_BASE + 1] and mask[A_BUY_BASE + 2]
    for p in range(BOARD_SIZE):
        assert not mask[A_SELL_BASE + p]  # empty board
    # board empty → only the identity SELECT_FINAL_ORDER (= base) is legal
    assert mask[A_SELECT_ORDER_BASE]
    assert not any(mask[A_SELECT_ORDER_BASE + 1 : A_SELECT_ORDER_BASE + 24])


def test_legal_orders_match_board_size():
    env = MiniBGEnv(seed=0)
    env._state.players[0].board = [make_minion("recruit"), make_minion("guard")]
    mask = env.legal_actions_mask
    expected = set(legal_order_indices(2))
    actual = {
        a - A_SELECT_ORDER_BASE
        for a in range(A_SELECT_ORDER_BASE, NUM_ENV_ACTIONS)
        if mask[a]
    }
    assert actual == expected
    assert len(actual) == 2


def test_step_buy_action_advances_state():
    env = MiniBGEnv(seed=0)
    _set_shop(env, 0, "recruit", "recruit", "recruit")
    result = env.step(A_BUY_BASE)
    assert result.terminated is False
    assert result.reward == 0.0
    assert env._state.players[0].board[0].card_id == "recruit"
    assert env._state.current_player_index == 0  # still my turn
    assert env._state.players[0].shop_actions_used == 1


def test_step_invalid_action_returns_penalty_and_does_not_mutate():
    env = MiniBGEnv(seed=0)
    env._state.players[0].gold = 0  # level up illegal
    snapshot = env.get_state_hash()
    result = env.step(A_LEVEL_UP)
    assert result.reward == INVALID_ACTION_REWARD
    assert result.terminated is False
    assert result.info["invalid_action"] is True
    assert result.info["termination_reason"] == "illegal"
    assert env.get_state_hash() == snapshot


def test_step_out_of_range_action_is_invalid():
    env = MiniBGEnv(seed=0)
    result = env.step(NUM_ENV_ACTIONS)
    assert result.info["invalid_action"] is True
    assert result.reward == INVALID_ACTION_REWARD


def test_select_final_order_reorders_board_and_finishes():
    env = MiniBGEnv(seed=0)
    env._state.players[0].board = [make_minion("recruit"), make_minion("guard")]
    swap_idx = next(
        j for j, p in enumerate(PERMUTATIONS_4) if p == (1, 0, 2, 3)
    )
    result = env.step(A_SELECT_ORDER_BASE + swap_idx)
    assert env._state.players[0].shopping_finished is True
    assert env._state.current_player_index == 1
    assert [m.card_id for m in env._state.players[0].board] == ["guard", "recruit"]
    assert result.terminated is False


def test_select_final_order_with_identity_when_empty_board():
    env = MiniBGEnv(seed=0)
    result = env.step(A_SELECT_ORDER_BASE)
    assert env._state.players[0].shopping_finished is True
    assert env._state.current_player_index == 1
    assert result.terminated is False


def test_select_final_order_invalid_perm_for_partial_board():
    env = MiniBGEnv(seed=0)
    env._state.players[0].board = [make_minion("recruit")]
    bad_idx = next(
        j for j, p in enumerate(PERMUTATIONS_4) if p == (0, 1, 3, 2)
    )
    result = env.step(A_SELECT_ORDER_BASE + bad_idx)
    assert result.info["invalid_action"] is True


def test_battle_resolution_updates_last_seen_and_signed():
    env = MiniBGEnv(seed=0)
    env._state.players[0].board = []
    env._state.players[1].board = [make_minion("big_guy")]
    env.step(A_SELECT_ORDER_BASE)  # P0 finishes empty
    result = env.step(A_SELECT_ORDER_BASE)  # P1 finishes → battle
    if env.done:
        assert result.terminated is True
    else:
        assert env._state.round_number == 2
    assert env._last_seen_enemy_board[0]  # P0 saw P1's board
    assert env._last_seen_enemy_board[1] == []  # P1 saw P0's empty board
    assert env._last_battle_signed[0] < 0  # P0 took damage
    assert env._last_battle_signed[1] > 0  # P1 dealt damage


def test_terminal_reward_for_winner_and_loser():
    env = MiniBGEnv(seed=0)
    env._state.players[0].health = 1
    env._state.players[0].board = []
    env._state.players[1].board = [make_minion("big_guy")]
    env.step(A_SELECT_ORDER_BASE)  # P0 finish
    result = env.step(A_SELECT_ORDER_BASE)  # P1 finish → battle, P0 dies
    assert result.terminated is True
    assert result.info["winner"] == -1
    assert result.info["termination_reason"] == "win"
    assert result.reward == 1.0


def test_terminal_reward_for_acting_player_when_loses():
    env = MiniBGEnv(seed=0)
    env._state.players[1].health = 1
    env._state.players[1].board = []
    env._state.players[0].board = [make_minion("big_guy")]
    env.step(A_SELECT_ORDER_BASE)  # P0 finishes; battle later
    result = env.step(A_SELECT_ORDER_BASE)  # P1 acts and is the loser
    assert result.terminated is True
    assert result.info["winner"] == 1
    assert result.info["termination_reason"] == "loss"
    assert result.reward == -1.0


def test_step_after_done_raises():
    env = MiniBGEnv(seed=0)
    env._state.players[0].health = 1
    env._state.players[0].board = []
    env._state.players[1].board = [make_minion("big_guy")]
    env.step(A_SELECT_ORDER_BASE)
    env.step(A_SELECT_ORDER_BASE)
    assert env.done is True
    with pytest.raises(ValueError):
        env.step(A_SELECT_ORDER_BASE)


def test_reset_clears_battle_history_and_seed_determinism():
    env = MiniBGEnv(seed=42)
    env._last_battle_signed = [0.5, -0.5]
    env._last_seen_enemy_board = [[make_minion("recruit")], [make_minion("guard")]]
    env.reset(seed=42)
    assert env._last_battle_signed == [0.0, 0.0]
    assert env._last_seen_enemy_board == [[], []]


def test_seeded_envs_produce_identical_initial_observations():
    env_a = MiniBGEnv(seed=123)
    env_b = MiniBGEnv(seed=123)
    obs_a = env_a.reset(seed=123)
    obs_b = env_b.reset(seed=123)
    assert np.array_equal(obs_a, obs_b)


def test_run_full_random_episode_terminates():
    env = MiniBGEnv(seed=7)
    rng = np.random.default_rng(7)
    steps = 0
    while not env.done and steps < 5000:
        mask = env.legal_actions_mask
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            break
        a = int(rng.choice(legal))
        env.step(a)
        steps += 1
    assert env.done
    assert steps < 5000


def test_obs_after_step_remains_correct_shape():
    env = MiniBGEnv(seed=0)
    obs = env.step(A_SELECT_ORDER_BASE).obs
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32


def test_action_cap_forces_finish_via_finish_action():
    env = MiniBGEnv(seed=0)
    env._state.players[0].gold = 1000
    for _ in range(MAX_SHOP_ACTIONS):
        env.step(A_ROLL)
    assert env._state.players[0].shopping_finished is True
    assert env._state.current_player_index == 1
