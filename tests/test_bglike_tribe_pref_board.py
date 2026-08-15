"""Per-round board form of the tribe-preference shaping.

Pays for what stands on the board when a round ends, every round it stays, and
once per minion — as opposed to ``tribe_pref_shaping``, which pays once at the
moment of purchase and then stops caring what happens to the card.
"""

from types import SimpleNamespace

import pytest

from src.bg_catalog.cards import Race
from src.training.bglike_perspective import BGLikeAgentPerspectiveEnv


class _M:
    def __init__(self, race):
        self.race = race


class _Seat:
    def __init__(self, pref):
        self.board = []
        self.tribe_pref = pref


def _env(pref=(1.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0), coef=0.025, seats=2):
    """An instance without __init__ — only the shaping fields are exercised."""
    env = BGLikeAgentPerspectiveEnv.__new__(BGLikeAgentPerspectiveEnv)
    env._tribe_pref_board_coef = coef
    env._tribe_pref_round_paid = {}
    env._bg_base = SimpleNamespace(
        state=SimpleNamespace(players=[_Seat(pref) for _ in range(seats)], round_number=1)
    )
    return env


def _pay(env, seat):
    return env._tribe_pref_board_reward({"acting_seat": seat})


def test_pays_superlinearly_in_the_stack():
    env = _env()
    env._bg_base.state.players[0].board = [_M(Race.BEAST), _M(Race.BEAST)]
    assert _pay(env, 0) == pytest.approx(0.025 * 2 ** 1.5)


def test_concentration_beats_spreading_at_equal_board_size():
    """Three of one tribe outscore one each of three, which is the whole point
    of the exponent."""
    one = _env()
    one._bg_base.state.players[0].board = [_M(Race.BEAST)] * 3
    many = _env()
    many._bg_base.state.players[0].board = [_M(Race.BEAST), _M(Race.MECHANICAL), _M(Race.MURLOC)]
    # pref: BEAST 1.0, MECHANICAL 0.5, MURLOC 0.0. One call each — the round key
    # makes a second call in the same round return 0.
    stacked, spread = _pay(one, 0), _pay(many, 0)
    assert stacked == pytest.approx(0.025 * 3 ** 1.5)
    assert spread == pytest.approx(0.025 * 1.5)
    assert stacked > spread


def test_the_stack_is_capped():
    env = _env()
    env._bg_base.state.players[0].board = [_M(Race.BEAST)] * 7
    assert _pay(env, 0) == pytest.approx(0.025 * 5 ** 1.5)


def test_a_disliked_tribe_costs():
    env = _env()
    env._bg_base.state.players[0].board = [_M(Race.DEMON)]
    assert _pay(env, 0) == pytest.approx(-0.025)


def test_mixed_board_sums_signed():
    env = _env()
    env._bg_base.state.players[0].board = [_M(Race.BEAST), _M(Race.DEMON), _M(Race.MECHANICAL)]
    # one each: 1.0 - 1.0 + 0.5 = 0.5
    assert _pay(env, 0) == pytest.approx(0.025 * 0.5)


def test_a_disliked_stack_costs_superlinearly_too():
    env = _env()
    env._bg_base.state.players[0].board = [_M(Race.DEMON)] * 3
    assert _pay(env, 0) == pytest.approx(-0.025 * 3 ** 1.5)


def test_paid_once_per_round_however_often_it_fires():
    env = _env()
    env._bg_base.state.players[0].board = [_M(Race.BEAST)]
    assert _pay(env, 0) == pytest.approx(0.025)
    assert _pay(env, 0) == 0.0
    assert _pay(env, 0) == 0.0


def test_the_next_round_pays_again():
    env = _env()
    env._bg_base.state.players[0].board = [_M(Race.BEAST)]
    assert _pay(env, 0) == pytest.approx(0.025)
    env._bg_base.state.round_number = 2
    assert _pay(env, 0) == pytest.approx(0.025)


def test_keeping_it_longer_pays_more_than_buying_and_selling():
    env = _env()
    p = env._bg_base.state.players[0]
    p.board = [_M(Race.BEAST)]
    total = 0.0
    for rnd in range(1, 5):
        env._bg_base.state.round_number = rnd
        total += _pay(env, 0)
    assert total == pytest.approx(4 * 0.025)


def test_a_minion_sold_before_the_round_ends_pays_nothing():
    env = _env()
    env._bg_base.state.players[0].board = []
    assert _pay(env, 0) == 0.0


def test_tribeless_minions_pay_nothing():
    env = _env()
    env._bg_base.state.players[0].board = [_M(None)]
    assert _pay(env, 0) == 0.0


def test_credit_is_per_seat():
    env = _env()
    env._bg_base.state.players[0].board = [_M(Race.BEAST)]
    env._bg_base.state.players[1].board = [_M(Race.BEAST)]
    assert _pay(env, 0) == pytest.approx(0.025)
    assert _pay(env, 1) == pytest.approx(0.025)


def test_pays_without_any_combat_flag():
    """Binding to combat_advanced skipped three rounds in four; the round key is
    the only thing that may gate the payment."""
    env = _env()
    env._bg_base.state.players[0].board = [_M(Race.BEAST)]
    paid = []
    for rnd in range(1, 6):
        env._bg_base.state.round_number = rnd
        paid.append(env._tribe_pref_board_reward({"acting_seat": 0}))
    assert paid == [pytest.approx(0.025)] * 5


def test_disabled_by_default():
    env = _env(coef=0.0)
    env._bg_base.state.players[0].board = [_M(Race.BEAST)]
    assert _pay(env, 0) == 0.0
