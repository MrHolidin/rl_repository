"""Defender-style adjacent battlecry — RL env decomposition only."""

from tests.conftest import obs_encode_pending_choice as encode_pending_choice
from tests.minibg_helpers import make_minion
from src.bg_core.effects import Keyword
from src.envs.minibg.env import MiniBGEnv
from src.envs.minibg.obs import PENDING_EFFECT_TAUNT_OFFSET, PENDING_IS_APPLY_OFFSET
from src.envs.minibg.structured_actions import StructActionType


def test_argus_flat_mask_includes_skip_after_first_pick():
    env = MiniBGEnv(seed=7)
    env.reset()
    idx = env._state.current_player_index
    p = env._state.players[idx]
    p.board = [make_minion("recruit"), make_minion("guard")]
    p.hand[0] = make_minion("defender_argus")
    from src.envs.minibg.action_map import A_APPLY_EFFECT_SKIP, A_PLACE_BASE, A_TARGET_BOARD_BASE

    env.step(A_PLACE_BASE)
    assert not env.legal_actions_mask[A_APPLY_EFFECT_SKIP]
    env.step(A_TARGET_BOARD_BASE + 1)
    assert env.legal_actions_mask[A_APPLY_EFFECT_SKIP]
    env.step(A_APPLY_EFFECT_SKIP)
    assert env._rl_pending is None
    assert len(env._state.players[idx].board) == 3


def test_argus_structured_place_then_apply_skip_for_single_neighbor():
    env = MiniBGEnv(seed=0)
    env.reset()
    idx = env._state.current_player_index
    p = env._state.players[idx]
    p.board = [make_minion("recruit"), make_minion("guard")]
    p.hand[0] = make_minion("defender_argus")
    from src.envs.minibg.action_map import A_PLACE_BASE

    env.step(A_PLACE_BASE)
    legal = env.legal_structured_actions()
    assert all(
        a.type in (StructActionType.APPLY_EFFECT, StructActionType.APPLY_EFFECT_SKIP)
        for a in legal
    )
    assert {a.args[0] for a in legal if a.type == StructActionType.APPLY_EFFECT} == {
        0,
        1,
    }

    pend = encode_pending_choice(p, rl_pending=env._rl_pending)
    assert pend[PENDING_IS_APPLY_OFFSET] == 1.0
    assert pend[PENDING_EFFECT_TAUNT_OFFSET] == 1.0

    pick = next(a for a in legal if a.args[0] == 1)
    env.step_structured(pick)
    assert env._rl_pending is not None
    skip = next(
        a for a in env.legal_structured_actions() if a.type == StructActionType.APPLY_EFFECT_SKIP
    )
    env.step_structured(skip)

    b = env._state.players[idx].board
    assert env._rl_pending is None
    assert len(b) == 3
    assert b[0].card_id != "EX1_093"
    assert b[1].bonus_attack == 1 and Keyword.TAUNT in b[1].keywords
    assert b[2].card_id == "EX1_093"


def test_game_argus_auto_buffs_both_neighbors():
    from src.bg_recruitment.targeted_battlecry import apply_targeted_on_place_battlecries
    from src.envs.minibg.game import MiniBGGame

    g = MiniBGGame(seed=1)
    s = g.initial_state()
    argus = make_minion("defender_argus")
    s.players[0].board = [make_minion("recruit"), argus, make_minion("guard")]
    apply_targeted_on_place_battlecries(
        g._shop_triggers, s.players[0], argus, rng=g._rng
    )
    b = s.players[0].board
    assert b[0].bonus_attack == 1 and Keyword.TAUNT in b[0].keywords
    assert b[2].bonus_attack == 1 and Keyword.TAUNT in b[2].keywords


def test_env_argus_two_neighbors_two_apply_then_between():
    env = MiniBGEnv(seed=3)
    env.reset()
    idx = env._state.current_player_index
    p = env._state.players[idx]
    p.board = [make_minion("recruit"), make_minion("guard")]
    p.hand[0] = make_minion("defender_argus")
    from src.envs.minibg.action_map import A_PLACE_BASE

    env.step(A_PLACE_BASE)
    assert env._rl_pending is not None

    legal = env.legal_structured_actions()
    env.step_structured(next(a for a in legal if a.args[0] == 0))
    legal2 = env.legal_structured_actions()
    env.step_structured(next(a for a in legal2 if a.args[0] == 1))

    b = env._state.players[idx].board
    assert env._rl_pending is None
    assert len(b) == 3
    assert b[1].card_id == "EX1_093"
    assert b[0].bonus_attack == 1 and Keyword.TAUNT in b[0].keywords
    assert b[2].bonus_attack == 1 and Keyword.TAUNT in b[2].keywords
