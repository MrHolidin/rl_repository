import numpy as np
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
)
from src.envs.minibg.actions import (
    BOARD_SIZE,
    HAND_SIZE,
    MAX_SHOP_ACTIONS,
    SHOP_SIZE,
)
from src.envs.minibg.cards import make_minion
from src.envs.reward_config import RewardConfig
from src.envs.minibg.env import INVALID_ACTION_REWARD, MiniBGEnv
from src.envs.minibg.obs import OBS_DIM
from src.envs.minibg.state import PlayerPhase
from src.registry import make_game


def _set_shop(env: MiniBGEnv, idx: int, *card_ids):
    p = env._state.players[idx]
    p.shop = [make_minion(cid) if cid is not None else None for cid in card_ids]
    while len(p.shop) < SHOP_SIZE:
        p.shop.append(None)


def _submit_identity_order(env: MiniBGEnv) -> None:
    if env.legal_actions_mask[A_FINISH]:
        env.step(A_FINISH)
    env.step(A_SELECT_ORDER_BASE)


def test_reset_returns_obs_of_correct_shape():
    env = MiniBGEnv(seed=0)
    obs = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32


def test_legal_actions_mask_is_phase_aware():
    env = MiniBGEnv(seed=0)
    mask = env.legal_actions_mask
    # Shop phase: shop actions legal, no SELECT_ORDER.
    assert mask[A_ROLL]
    assert mask[A_FINISH]
    assert all(mask[A_BUY_BASE + i] for i in range(SHOP_SIZE))
    assert not any(mask[A_SELECT_ORDER_BASE + j] for j in range(NUM_PERMS))
    # Order phase with empty board: only the identity perm is legal (k=0 -> 1
    # canonical representative).
    env.step(A_FINISH)
    mask = env.legal_actions_mask
    legal_order = [j for j in range(NUM_PERMS) if mask[A_SELECT_ORDER_BASE + j]]
    assert legal_order == [0]
    assert not mask[A_FINISH] and not mask[A_ROLL]
    assert not any(mask[A_BUY_BASE + i] for i in range(SHOP_SIZE))


def test_legal_select_order_count_matches_k_factorial():
    """ORDER phase exposes exactly ``k!`` SELECT_ORDER actions for board size k."""
    import math

    env = MiniBGEnv(seed=0)
    for k in range(BOARD_SIZE + 1):
        env.reset(seed=0)
        env._state.players[0].board = [make_minion("recruit") for _ in range(k)]
        env.step(A_FINISH)
        legal = [j for j in range(NUM_PERMS) if env.legal_actions_mask[A_SELECT_ORDER_BASE + j]]
        assert len(legal) == max(1, math.factorial(k)), f"k={k}: expected {max(1, math.factorial(k))} got {len(legal)}"
        # Every legal perm must be canonical: tail beyond k is identity.
        for j in legal:
            tail = PERMUTATIONS_4[j][k:]
            assert tail == tuple(range(k, BOARD_SIZE)), f"k={k}: non-canonical perm at j={j}"


def test_buy_to_hand_then_place_to_board():
    env = MiniBGEnv(seed=0)
    _set_shop(env, 0, "recruit", "recruit", "recruit")
    env.step(A_BUY_BASE)
    p = env._state.players[0]
    assert p.hand[0] is not None and p.hand[0].card_id == "recruit"
    assert p.board == []
    assert env._state.current_player_index == 0  # BUY does not pass turn
    env.step(A_PLACE_BASE)
    p = env._state.players[0]  # apply_action returned a copied state
    assert p.hand[0] is None
    assert [m.card_id for m in p.board] == ["recruit"]


def test_non_canonical_select_order_is_illegal_for_partial_board():
    env = MiniBGEnv(seed=0)
    env._state.players[0].board = [make_minion("recruit")]
    env.step(A_FINISH)
    # k=1: only identity perm is canonical; a perm whose tail differs is illegal.
    weird_idx = next(j for j, p in enumerate(PERMUTATIONS_4) if p == (3, 0, 2, 1))
    snapshot = env.get_state_hash()
    res = env.step(A_SELECT_ORDER_BASE + weird_idx)
    assert res.info["invalid_action"] is True
    assert env.get_state_hash() == snapshot
    # Identity perm finalises the order phase.
    res = env.step(A_SELECT_ORDER_BASE)
    assert res.info["invalid_action"] is False
    assert env._state.players[0].phase == PlayerPhase.DONE
    assert [m.card_id for m in env._state.players[0].board] == ["recruit"]


def test_action_budget_exhaustion_auto_flips_phase():
    env = MiniBGEnv(seed=0)
    env._state.players[0].gold = 1000
    for _ in range(MAX_SHOP_ACTIONS):
        env.step(A_ROLL)
    p = env._state.players[0]
    assert p.phase == PlayerPhase.ORDER
    assert env._state.current_player_index == 0  # turn stays until SELECT_ORDER


def test_select_order_in_shop_phase_is_illegal():
    env = MiniBGEnv(seed=0)
    snapshot = env.get_state_hash()
    res = env.step(A_SELECT_ORDER_BASE)
    assert res.info["invalid_action"] is True
    assert env.get_state_hash() == snapshot


def test_terminal_reward_for_winner():
    env = MiniBGEnv(seed=0)
    env._state.players[0].health = 1
    env._state.players[0].board = []
    env._state.players[1].board = [make_minion("big_guy")]
    _submit_identity_order(env)
    env.step(A_FINISH)
    res = env.step(A_SELECT_ORDER_BASE)
    assert res.terminated and res.info["winner"] == -1
    assert res.info["termination_reason"] == "win"
    assert res.reward == 1.0


def test_run_full_random_episode_terminates():
    env = MiniBGEnv(seed=7)
    rng = np.random.default_rng(7)
    steps = 0
    while not env.done and steps < 5000:
        legal = np.flatnonzero(env.legal_actions_mask)
        env.step(int(rng.choice(legal)))
        steps += 1
    assert env.done
    assert steps < 5000


def test_make_game_passes_battle_damage_shaping():
    env = make_game("minibg", seed=0, battle_damage_shaping=0.2)
    assert env._battle_damage_shaping == pytest.approx(0.2)


def test_reward_config_invalid_action_override():
    rc = RewardConfig(invalid_action=-2.0)
    env = MiniBGEnv(seed=0, reward_config=rc)
    env.reset(seed=0)
    r = env.step(NUM_ENV_ACTIONS)
    assert r.info.get("invalid_action") is True
    assert r.reward == -2.0
