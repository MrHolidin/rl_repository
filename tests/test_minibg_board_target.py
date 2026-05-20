"""Targeted friendly battlecry (+1/+1 to chosen board minion) — RL env only."""

from src.bg_catalog.cards import make_minion
from src.envs.minibg.env import MiniBGEnv
from src.envs.minibg.structured_actions import StructActionType


def test_single_other_minion_auto_buffs_without_modal():
    env = MiniBGEnv(seed=0)
    env.reset()
    p = env._state.players[0]
    p.board = [make_minion("recruit")]
    p.hand[0] = make_minion("target_buffer")
    from src.envs.minibg.action_map import A_PLACE_BASE

    env.step(A_PLACE_BASE)
    pick = next(a for a in env.legal_structured_actions() if a.type == StructActionType.APPLY_EFFECT)
    env.step_structured(pick)
    p0 = env._state.players[0]
    assert env._rl_pending is None
    assert len(p0.board) == 2
    assert p0.board[0].bonus_attack == 1 and p0.board[0].bonus_health == 1
    assert p0.board[1].bonus_attack == 0


def test_two_minions_opens_rl_apply_then_buffs_chosen():
    env = MiniBGEnv(seed=1)
    env.reset()
    p = env._state.players[0]
    p.board = [make_minion("recruit"), make_minion("guard")]
    p.hand[0] = make_minion("target_buffer")
    from src.envs.minibg.action_map import A_PLACE_BASE

    env.step(A_PLACE_BASE)
    assert env._rl_pending is not None
    legal = env.legal_structured_actions()
    assert all(a.type == StructActionType.APPLY_EFFECT for a in legal)
    assert {a.args[0] for a in legal} == {0, 1}

    pick = next(a for a in legal if a.args[0] == 1)
    env.step_structured(pick)
    p0 = env._state.players[0]
    assert env._rl_pending is None
    assert p0.board[1].bonus_attack == 1 and p0.board[1].bonus_health == 1
    assert p0.board[0].bonus_attack == 0


def test_env_target_board_pick():
    env = MiniBGEnv(seed=2)
    env.reset()
    env._state.players[0].board = [make_minion("recruit"), make_minion("guard")]
    env._state.players[0].hand[0] = make_minion("target_buffer")
    from src.envs.minibg.action_map import A_PLACE_BASE, A_TARGET_BOARD_BASE

    env.step(A_PLACE_BASE)
    assert env._rl_pending is not None
    mask = env.legal_actions_mask
    assert mask[A_TARGET_BOARD_BASE]
    assert mask[A_TARGET_BOARD_BASE + 1]
    env.step(A_TARGET_BOARD_BASE)
    assert env._rl_pending is None
    assert env._state.players[0].board[0].bonus_attack == 1


def test_brann_buff_target_twice_on_one_pick():
    from src.bg_recruitment.targeted_battlecry import apply_targeted_on_place_battlecries
    from src.envs.minibg.game import MiniBGGame

    g = MiniBGGame(seed=4)
    s = g.initial_state()
    p = s.players[0]
    recruit = make_minion("recruit")
    p.board = [recruit, make_minion("brann")]
    buffer = make_minion("target_buffer")
    p.board.append(buffer)
    apply_targeted_on_place_battlecries(
        g._shop_triggers,
        p,
        buffer,
        rng=g._rng,
        forced_buff_target=recruit,
    )
    assert recruit.bonus_attack == 2 and recruit.bonus_health == 2


def test_rl_brann_one_apply_buffs_twice():
    env = MiniBGEnv(seed=5)
    env.reset()
    idx = env._state.current_player_index
    p = env._state.players[idx]
    recruit = make_minion("recruit")
    p.board = [recruit, make_minion("brann"), make_minion("guard")]
    p.hand[0] = make_minion("target_buffer")
    from src.envs.minibg.action_map import A_PLACE_BASE

    env.step(A_PLACE_BASE)
    env.step_structured(
        next(
            a
            for a in env.legal_structured_actions()
            if a.type == StructActionType.APPLY_EFFECT and a.args[0] == 0
        )
    )
    assert env._rl_pending is None
    assert recruit.bonus_attack == 2 and recruit.bonus_health == 2


def test_finish_blocked_while_rl_effect_pending():
    env = MiniBGEnv(seed=3)
    env.reset()
    env._state.players[0].board = [make_minion("recruit"), make_minion("guard")]
    env._state.players[0].hand[0] = make_minion("target_buffer")
    from src.envs.minibg.action_map import A_PLACE_BASE, A_FINISH

    env.step(A_PLACE_BASE)
    assert env._rl_pending is not None
    assert not env.legal_actions_mask[A_FINISH]
