"""Round increment, combat leaves shop boards unchanged, ON_TURN_START ordering."""

import numpy as np

from tests.minibg_helpers import make_minion
from src.bg_core.effects import Ability, BuffSelf, Trigger
from tests.minibg_helpers import simulate_battle
from src.envs.minibg.game import MiniBGGame


def test_resolve_battle_preserves_shop_boards():
    g = MiniBGGame(seed=0)
    s = g.initial_state()
    before = make_minion("recruit")
    s.players[0].board = [before]
    s.players[1].board = []
    g._resolve_battle_and_advance(s)
    assert s.players[0].board == [before]
    assert [m.card_id for m in s.players[0].board] == ["EX1_162"]
    assert s.players[1].board == []


def test_micro_machine_turn_start_after_round_increment():
    g = MiniBGGame(seed=0)
    s = g.initial_state()
    s.players[0].board = [make_minion("micro_machine")]
    s.players[1].board = []
    g._resolve_battle_and_advance(s)
    assert s.round_number == 2
    mm = s.players[0].board[0]
    assert mm.card_id == "BGS_027"
    assert mm.raw_attack == 2


def test_turn_start_board_left_to_right_two_sources():
    g = MiniBGGame(seed=1)
    s = g.initial_state()
    m1 = make_minion("recruit")
    m2 = make_minion("recruit")
    m1.abilities = (Ability(Trigger.ON_TURN_START, BuffSelf(attack=1, health=0)),)
    m2.abilities = (Ability(Trigger.ON_TURN_START, BuffSelf(attack=1, health=0)),)
    s.players[0].board = [m1, m2]
    s.players[1].board = []
    g._resolve_battle_and_advance(s)
    assert [m.raw_attack for m in s.players[0].board] == [3, 3]


def test_turn_start_hand_slot_runs_after_board():
    g = MiniBGGame(seed=0)
    s = g.initial_state()
    hb = make_minion("recruit")
    hb.abilities = (Ability(Trigger.ON_TURN_START, BuffSelf(attack=5, health=0)),)
    s.players[0].board = [make_minion("micro_machine")]
    s.players[0].hand[0] = hb
    s.players[1].board = []
    g._resolve_battle_and_advance(s)
    assert s.players[0].hand[0] is hb
    assert hb.raw_attack == 7
    assert s.players[0].board[0].raw_attack == 2


def test_khadgar_tokens_on_persistent_board():
    p0 = [make_minion("pack_rat"), make_minion("khadgar")]
    p1 = [make_minion("recruit")]
    b0: list = []
    simulate_battle(
        p0,
        p1,
        p0_has_initiative=True,
        rng=np.random.default_rng(0),
        p0_board_out=b0,
    )
    assert sum(1 for m in b0 if m.card_id == "CFM_316t") == 4
    assert any(m.card_id == "DAL_575" for m in b0)
