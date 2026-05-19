"""Magnetic merge legality and HS-style combine semantics."""

from src.envs.minibg.action_map import A_MAGNET_BASE
from src.envs.minibg.actions import magnet_game_action
from src.bg_catalog.cards import make_minion
from src.bg_core.effects import Keyword, Trigger
from src.envs.minibg.env import MiniBGEnv
from src.envs.minibg.game import MiniBGGame


def test_magnet_illegal_without_mech_target():
    g = MiniBGGame(seed=0)
    s = g.initial_state()
    p = s.players[0]
    p.board = [make_minion("recruit")]
    p.hand[0] = make_minion("annoy_o_module")
    assert int(magnet_game_action(0, 0)) not in set(g.legal_actions(s))


def test_magnet_illegal_non_magnetic_in_hand():
    g = MiniBGGame(seed=0)
    s = g.initial_state()
    p = s.players[0]
    p.board = [make_minion("toy_mech")]
    p.hand[0] = make_minion("toy_mech")
    assert int(magnet_game_action(0, 0)) not in set(g.legal_actions(s))


def test_magnet_merges_stats_dr_order_and_strips_magnetic():
    g = MiniBGGame(seed=0)
    s = g.initial_state()
    p = s.players[0]
    p.board = [make_minion("mech_base_dr")]
    p.hand[0] = make_minion("magnetic_dr_rat")
    ma = int(magnet_game_action(0, 0))
    assert ma in set(g.legal_actions(s))
    s2 = g.apply_action(s, ma)
    m = s2.players[0].board[0]
    assert m.card_id == "EX1_556"
    assert m.raw_attack == 5
    assert m.max_health == 4
    drs = [ab for ab in m.abilities if ab.trigger == Trigger.ON_DEATH]
    assert len(drs) == 2
    assert drs[0].effect.token_id == "skele21"
    assert drs[1].effect.token_id == "BOT_312t"
    assert Keyword.MAGNETIC not in m.all_keywords


def test_magnet_preserves_board_buffs():
    g = MiniBGGame(seed=0)
    s = g.initial_state()
    p = s.players[0]
    t = make_minion("toy_mech")
    t.bonus_attack = 5
    p.board = [t]
    p.hand[0] = make_minion("annoy_o_module")
    s2 = g.apply_action(s, int(magnet_game_action(0, 0)))
    m = s2.players[0].board[0]
    assert m.bonus_attack == 5
    assert m.raw_attack == 8
    assert m.max_health == 5


def test_golden_magnetic_adds_double_stats_but_not_golden_identity():
    """BG-style: magnet stats from a golden module apply; merged minion stays its card identity (not golden)."""
    g = MiniBGGame(seed=0)
    s = g.initial_state()
    p = s.players[0]
    p.board = [make_minion("toy_mech")]
    p.hand[0] = make_minion("annoy_o_module_golden")
    s2 = g.apply_action(s, int(magnet_game_action(0, 0)))
    m = s2.players[0].board[0]
    assert not m.is_golden
    assert m.raw_attack == 5
    assert m.max_health == 9


def test_annoy_module_adds_taunt_divine_shield():
    g = MiniBGGame(seed=0)
    s = g.initial_state()
    p = s.players[0]
    p.board = [make_minion("toy_mech")]
    p.hand[0] = make_minion("annoy_o_module")
    s2 = g.apply_action(s, int(magnet_game_action(0, 0)))
    m = s2.players[0].board[0]
    assert Keyword.TAUNT in m.all_keywords
    assert m.has_shield


def test_env_mask_includes_magnet_when_legal():
    env = MiniBGEnv(seed=0)
    p = env.state.players[0]
    p.board = [make_minion("toy_mech")]
    p.hand[0] = make_minion("annoy_o_module")
    assert env.legal_actions_mask[A_MAGNET_BASE]
