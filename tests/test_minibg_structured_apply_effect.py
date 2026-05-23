"""Structured APPLY_EFFECT for targeted effect modals (RL env)."""

from tests.conftest import obs_encode_pending_choice as encode_pending_choice
from tests.minibg_helpers import make_minion
from src.envs.minibg.env import MiniBGEnv
from src.envs.minibg.obs import (
    OBS_DIM,
    PENDING_APPLY_REMAINING_OFFSET,
    PENDING_EFFECT_ATK_OFFSET,
    PENDING_EFFECT_HP_OFFSET,
    PENDING_HEADER_OFFSET,
    PENDING_IS_APPLY_OFFSET,
    PENDING_ELIGIBLE_OFFSET,
    PENDING_PICKED_OFFSET,
)
from src.envs.minibg.structured_actions import StructAction, StructActionType


def test_obs_encodes_apply_effect_modal():
    env = MiniBGEnv(seed=0)
    env.reset()
    p = env._state.players[0]
    p.board = [make_minion("recruit"), make_minion("guard")]
    p.hand[0] = make_minion("target_buffer")
    from src.envs.minibg.action_map import A_PLACE_BASE

    env.step(A_PLACE_BASE)
    p = env._state.players[0]
    pend = encode_pending_choice(p, rl_pending=env._rl_pending)
    assert pend[PENDING_IS_APPLY_OFFSET] == 1.0
    assert pend[PENDING_EFFECT_ATK_OFFSET] > 0
    assert pend[PENDING_EFFECT_HP_OFFSET] > 0
    assert pend[PENDING_ELIGIBLE_OFFSET] == 1.0
    assert pend[PENDING_ELIGIBLE_OFFSET + 1] == 1.0


def test_structured_legal_apply_effect_tokens():
    env = MiniBGEnv(seed=1)
    env.reset()
    p = env._state.players[0]
    p.board = [make_minion("recruit"), make_minion("guard")]
    p.hand[0] = make_minion("target_buffer")
    from src.envs.minibg.action_map import A_PLACE_BASE

    env.step(A_PLACE_BASE)
    legal = env.legal_structured_actions()
    assert all(a.type == StructActionType.APPLY_EFFECT for a in legal)
    assert all(len(a.args) == 1 for a in legal)
    assert {a.args[0] for a in legal} == {0, 1}


def test_structured_step_apply_effect():
    env = MiniBGEnv(seed=2)
    env.reset()
    env._state.players[0].board = [make_minion("recruit"), make_minion("guard")]
    env._state.players[0].hand[0] = make_minion("target_buffer")
    from src.envs.minibg.action_map import A_PLACE_BASE

    env.step(A_PLACE_BASE)
    legal = env.legal_structured_actions()
    pick = next(a for a in legal if a.args[0] == 0)
    res = env.step_structured(pick)
    assert not res.info.get("invalid_action")
    assert env._state.players[0].board[0].bonus_attack == 1
    assert env._rl_pending is None


def test_obs_dim_includes_apply_section():
    env = MiniBGEnv(seed=0)
    obs = env.reset()
    assert obs.shape == (OBS_DIM,)


def test_obs_apply_remaining_and_picked_mask():
    env = MiniBGEnv(seed=4)
    env.reset()
    idx = env._state.current_player_index
    p = env._state.players[idx]
    p.board = [make_minion("recruit"), make_minion("guard")]
    p.hand[0] = make_minion("defender_argus")
    from src.envs.minibg.action_map import A_PLACE_BASE

    env.step(A_PLACE_BASE)
    pend = encode_pending_choice(p, rl_pending=env._rl_pending)
    assert pend[PENDING_APPLY_REMAINING_OFFSET] == 1.0
    assert pend[PENDING_HEADER_OFFSET + 2] == 1.0
    assert pend[PENDING_PICKED_OFFSET : PENDING_PICKED_OFFSET + 4].sum() == 0.0

    env.step_structured(
        next(a for a in env.legal_structured_actions() if a.args[0] == 1)
    )
    pend2 = encode_pending_choice(p, rl_pending=env._rl_pending)
    assert pend2[PENDING_APPLY_REMAINING_OFFSET] == 0.5
    assert pend2[PENDING_PICKED_OFFSET + 1] == 1.0
    assert pend2[PENDING_PICKED_OFFSET + 0] == 0.0
