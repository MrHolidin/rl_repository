"""Bot action outside legal_actions_mask triggers RuntimeError and RL_RUN_DIR log."""

from pathlib import Path

import numpy as np
import pytest

from tests.minibg_helpers import make_minion
from src.envs.minibg.heuristic_bots.bots import Tier1RandomBot
from src.envs.minibg.heuristic_bots.common import masked_finish
from src.envs.minibg.invariants import assert_action_in_legal_mask
from src.envs.minibg.state import MiniBGState, PlayerPhase, PlayerState


def _shop_state() -> MiniBGState:
    p = PlayerState(
        health=40,
        gold=10,
        tavern_tier=3,
        next_tier_up_cost=5,
        board=[make_minion("recruit")],
        shop=[make_minion("recruit")] * 6,
        hand=[None] * 5,
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
    )
    return MiniBGState(
        players=(p, p),
        round_number=1,
        current_player_index=0,
        initiative_player=0,
        winner=None,
        done=False,
    )


def test_assert_action_not_in_mask_raises_and_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("RL_RUN_DIR", str(tmp_path))
    state = _shop_state()
    mask = np.zeros(80, dtype=bool)
    mask[3] = True
    with pytest.raises(RuntimeError, match="ILLEGAL_ACTION"):
        assert_action_in_legal_mask(state, 15, mask, where="test")
    log = Path(tmp_path) / "illegal_action_assertion.log"
    assert log.is_file()
    assert "ILLEGAL_ACTION" in log.read_text()


def test_masked_finish_uses_target_when_finish_masked():
    from src.envs.minibg.action_map import A_FINISH, A_TARGET_BOARD_0

    mask = np.zeros(80, dtype=bool)
    mask[A_TARGET_BOARD_0] = True
    assert masked_finish(mask) == A_TARGET_BOARD_0


def test_heuristic_choose_action_always_in_mask():
    """Smoke: scripted bot never returns an action outside env mask."""
    from src.envs.minibg.env import MiniBGEnv

    env = MiniBGEnv(seed=42)
    bot = Tier1RandomBot(seed=1)
    for step_i in range(200):
        if env.done:
            env.reset(seed=1000 + step_i)
        a = bot.choose_action(env)
        mask = env.legal_actions_mask
        assert 0 <= a < len(mask) and bool(mask[a]), f"illegal action {a}"
        env.step(a)
