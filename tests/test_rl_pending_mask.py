"""During ``_rl_pending``, flat mask must only expose TARGET / SKIP (no shop actions)."""

import pytest

from tests.minibg_helpers import make_minion
from src.envs.minibg.action_map import (
    A_BUY_BASE,
    A_FINISH,
    A_PLACE_BASE,
    A_TARGET_BOARD_0,
)
from src.envs.minibg.env import MiniBGEnv
from src.envs.minibg.rl_place import open_rl_place_plan


def _open_pending(env: MiniBGEnv) -> None:
    p = env._state.players[0]
    p.board = [make_minion("recruit"), make_minion("guard")]
    p.hand[0] = make_minion("target_buffer")
    env.step(A_PLACE_BASE)
    assert env._rl_pending is not None


def test_rl_pending_mask_excludes_shop_actions():
    env = MiniBGEnv(seed=0, patch_dir="data/bgcore/15_6_2_36393")
    env.reset(seed=0)
    _open_pending(env)
    mask = env.legal_actions_mask
    assert bool(mask[A_TARGET_BOARD_0])
    assert not bool(mask[A_BUY_BASE])
    assert not bool(mask[A_FINISH])


def test_buy_during_rl_pending_raises():
    env = MiniBGEnv(seed=1, patch_dir="data/bgcore/15_6_2_36393")
    env.reset(seed=1)
    _open_pending(env)
    with pytest.raises(RuntimeError, match="ILLEGAL_ACTION"):
        env.step(A_BUY_BASE)
