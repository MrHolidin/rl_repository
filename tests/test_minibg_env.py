import numpy as np
import pytest

from src.envs.minibg.action_map import (
    A_BUY_BASE,
    A_FINISH,
    A_FINISH_FREEZE_SHOP,
    A_LEVEL_UP,
    A_PLACE_BASE,
    A_ROLL,
    A_SELL_BASE,
    A_SWAP_BOARD_0,
    NUM_ENV_ACTIONS,
    NUM_SWAP_ADJ,
)
from src.envs.minibg.actions import (
    BOARD_SIZE,
    HAND_SIZE,
    MAX_SHOP_SLOTS,
    MAX_SHOP_ACTIONS,
    shop_offers_count,
)
from src.bg_catalog.cards import make_minion
from src.envs.reward_config import RewardConfig
from src.envs.minibg.env import INVALID_ACTION_REWARD, MiniBGEnv
from src.envs.minibg.obs import OBS_DIM
from src.envs.minibg.state import PlayerPhase
from src.registry import make_game


def _set_shop(env: MiniBGEnv, idx: int, *card_ids):
    p = env._state.players[idx]
    p.shop = [make_minion(cid) if cid is not None else None for cid in card_ids]
    while len(p.shop) < MAX_SHOP_SLOTS:
        p.shop.append(None)


def _submit_identity_order(env: MiniBGEnv) -> None:
    if env.legal_actions_mask[A_FINISH]:
        env.step(A_FINISH)


def test_reset_returns_obs_of_correct_shape():
    env = MiniBGEnv(seed=0)
    obs = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32


def test_legal_actions_mask_includes_swaps_during_shop():
    env = MiniBGEnv(seed=0)
    mask = env.legal_actions_mask
    assert mask[A_ROLL]
    assert mask[A_FINISH]
    assert mask[A_FINISH_FREEZE_SHOP]
    assert not any(mask[A_SWAP_BOARD_0 + i] for i in range(NUM_SWAP_ADJ))
    n_buy = shop_offers_count(env._state.players[0].tavern_tier)
    assert all(mask[A_BUY_BASE + i] for i in range(n_buy))
    assert not any(mask[A_BUY_BASE + i] for i in range(n_buy, MAX_SHOP_SLOTS))

    env._state.players[0].board = [
        make_minion("recruit"),
        make_minion("guard"),
    ]
    mask = env.legal_actions_mask
    assert mask[A_SWAP_BOARD_0]
    assert not mask[A_SWAP_BOARD_0 + 1]


def test_shop_phase_legal_swap_count():
    """SHOP: adjacent swaps among occupied positions (k−1 when k≥2), alongside shop actions."""
    env = MiniBGEnv(seed=0)
    for k in range(BOARD_SIZE + 1):
        env.reset(seed=0)
        env._state.players[0].board = [make_minion("recruit") for _ in range(k)]
        m = env.legal_actions_mask
        expected_swaps = 0 if k < 2 else k - 1
        for i in range(NUM_SWAP_ADJ):
            assert m[A_SWAP_BOARD_0 + i] == (i < expected_swaps), f"k={k} i={i}"


def test_buy_to_hand_then_place_to_board():
    env = MiniBGEnv(seed=0)
    _set_shop(env, 0, "recruit", "recruit", "recruit")
    env.step(A_BUY_BASE)
    p = env._state.players[0]
    assert p.hand[0] is not None and p.hand[0].card_id == "EX1_162"
    assert p.board == []
    assert env._state.current_player_index == 0
    env.step(A_PLACE_BASE)
    p = env._state.players[0]
    assert p.hand[0] is None
    assert [m.card_id for m in p.board] == ["EX1_162"]


def test_illegal_swap_for_single_minion_is_rejected():
    env = MiniBGEnv(seed=0)
    env._state.players[0].board = [make_minion("recruit")]
    snapshot = env.get_state_hash()
    res = env.step(A_SWAP_BOARD_0)
    assert res.info["invalid_action"] is True
    assert env.get_state_hash() == snapshot
    res = env.step(A_FINISH)
    assert res.info["invalid_action"] is False
    assert env._state.players[0].phase == PlayerPhase.DONE
    assert [m.card_id for m in env._state.players[0].board] == ["EX1_162"]


def test_action_budget_exhaustion_stays_in_shop():
    env = MiniBGEnv(seed=0)
    env._state.players[0].gold = 1000
    for _ in range(MAX_SHOP_ACTIONS):
        env.step(A_ROLL)
    p = env._state.players[0]
    assert p.phase == PlayerPhase.SHOP
    assert p.shop_actions_used >= MAX_SHOP_ACTIONS
    assert env.legal_actions_mask[A_FINISH]
    assert env._state.current_player_index == 0


def test_swap_action_with_empty_board_is_illegal():
    env = MiniBGEnv(seed=0)
    snapshot = env.get_state_hash()
    res = env.step(A_SWAP_BOARD_0)
    assert res.info["invalid_action"] is True
    assert env.get_state_hash() == snapshot


def test_terminal_reward_for_winner():
    env = MiniBGEnv(seed=0)
    env._state.players[0].health = 1
    env._state.players[0].board = []
    env._state.players[1].board = [make_minion("big_guy")]
    _submit_identity_order(env)
    res = env.step(A_FINISH)
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
