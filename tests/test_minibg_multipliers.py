import numpy as np

from src.envs.minibg.actions import Action
from tests.minibg_helpers import make_minion
from tests.minibg_helpers import simulate_battle
from src.envs.minibg.game import MiniBGGame


def test_brann_triples_vulgar_hero_hits():
    g = MiniBGGame(seed=0)
    s = g.initial_state()
    s.players[0].board = [make_minion("brann_golden")]
    s.players[0].hand[0] = make_minion("vulgar_homunculus")
    s2 = g.apply_action(s, int(Action.PLACE_HAND_0))
    assert s2.players[0].hero_damage_taken_total == 6


def test_two_branns_multiply_battlecries():
    g = MiniBGGame(seed=0)
    s = g.initial_state()
    s.players[0].board = [make_minion("brann"), make_minion("brann")]
    s.players[0].hand[0] = make_minion("vulgar_homunculus")
    s2 = g.apply_action(s, int(Action.PLACE_HAND_0))
    assert s2.players[0].hero_damage_taken_total == 8


def test_khadgar_doubles_pack_rat_deathrattle():
    p0 = [make_minion("pack_rat"), make_minion("khadgar")]
    p1 = [make_minion("recruit")]
    surv: list = []
    simulate_battle(
        p0, p1, p0_has_initiative=True, rng=np.random.default_rng(0), p0_survivors_out=surv
    )
    assert surv.count("CFM_316t") == 4


def test_baron_doubles_deathrattle_summons():
    p0 = [make_minion("pack_rat"), make_minion("baron_rivendare")]
    p1 = [make_minion("recruit")]
    surv: list = []
    simulate_battle(
        p0, p1, p0_has_initiative=True, rng=np.random.default_rng(0), p0_survivors_out=surv
    )
    assert surv.count("CFM_316t") == 4


def test_golden_baron_triples_deathrattle_summons():
    p0 = [make_minion("pack_rat"), make_minion("baron_golden")]
    p1 = [make_minion("recruit")]
    surv: list = []
    simulate_battle(
        p0, p1, p0_has_initiative=True, rng=np.random.default_rng(0), p0_survivors_out=surv
    )
    assert surv.count("CFM_316t") == 6


def test_golden_khadgar_triples_deathrattle_summons():
    p0 = [make_minion("pack_rat"), make_minion("khadgar_golden")]
    p1 = [make_minion("recruit")]
    surv: list = []
    simulate_battle(
        p0, p1, p0_has_initiative=True, rng=np.random.default_rng(0), p0_survivors_out=surv
    )
    assert surv.count("CFM_316t") == 6


def test_baron_and_khadgar_both_apply_under_board_cap():
    p0 = [
        make_minion("pack_rat"),
        make_minion("baron_rivendare"),
        make_minion("khadgar"),
    ]
    p1 = [make_minion("recruit")]
    surv: list = []
    simulate_battle(
        p0, p1, p0_has_initiative=True, rng=np.random.default_rng(0), p0_survivors_out=surv
    )
    # Pack rat summons two rats per DR; Baron x2 Khadgar x2 caps on a 7-slot board.
    assert surv.count("CFM_316t") == 5
