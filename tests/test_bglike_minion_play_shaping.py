"""Play shaping: pay the acting seat once per copy of a named minion it PLAYS.

Board only. Distinct from ``minions_shaping``, which scores the FINAL board
terminally, and from the DvD acquisition reward, which counts own-tribe minions
rather than a named card and only fires with a nonzero diversity coefficient.
"""

from types import SimpleNamespace

import pytest

from src.training.bglike_perspective import BGLikeAgentPerspectiveEnv

NOMI = "Nomi, Kitchen Nightmare"


class _Card:
    def __init__(self, name):
        self.name = name


class _Seat:
    def __init__(self):
        self.board = []
        self.hand = []


def _env(bonuses=None, seats=2):
    """An instance without __init__ — only the shaping fields are exercised."""
    env = BGLikeAgentPerspectiveEnv.__new__(BGLikeAgentPerspectiveEnv)
    env._play_shaping = dict(bonuses if bonuses is not None else {NOMI: 0.2})
    env._play_paid = {}
    env._bg_base = SimpleNamespace(
        state=SimpleNamespace(players=[_Seat() for _ in range(seats)])
    )
    return env


def _pay(env, seat):
    return env._minion_play_reward({"acting_seat": seat})


def test_holding_it_in_hand_pays_nothing():
    env = _env()
    env._bg_base.state.players[0].hand.append(_Card(NOMI))
    assert _pay(env, 0) == 0.0


def test_pays_when_it_reaches_the_board():
    env = _env()
    p = env._bg_base.state.players[0]
    p.hand.append(_Card(NOMI))
    assert _pay(env, 0) == 0.0
    p.hand.clear()
    p.board.append(_Card(NOMI))
    assert _pay(env, 0) == pytest.approx(0.2)
    # Still standing there is not playing it again.
    assert _pay(env, 0) == 0.0


def test_a_second_copy_pays_again():
    env = _env()
    p = env._bg_base.state.players[0]
    p.board.append(_Card(NOMI))
    assert _pay(env, 0) == pytest.approx(0.2)
    p.board.append(_Card(NOMI))
    assert _pay(env, 0) == pytest.approx(0.2)


def test_selling_and_replaying_cannot_farm_the_bonus():
    env = _env()
    p = env._bg_base.state.players[0]
    p.board.append(_Card(NOMI))
    assert _pay(env, 0) == pytest.approx(0.2)
    p.board.clear()
    assert _pay(env, 0) == 0.0
    p.board.append(_Card(NOMI))
    assert _pay(env, 0) == 0.0


def test_credit_is_per_seat():
    env = _env()
    env._bg_base.state.players[0].board.append(_Card(NOMI))
    env._bg_base.state.players[1].board.append(_Card(NOMI))
    assert _pay(env, 0) == pytest.approx(0.2)
    assert _pay(env, 1) == pytest.approx(0.2)


def test_other_minions_pay_nothing():
    env = _env()
    env._bg_base.state.players[0].board.append(_Card("Alleycat"))
    assert _pay(env, 0) == 0.0


def test_disabled_by_default():
    env = _env(bonuses={})
    env._bg_base.state.players[0].board.append(_Card(NOMI))
    assert _pay(env, 0) == 0.0


def test_unknown_name_fails_loudly():
    from src.training.bglike_perspective import _assert_minion_names_in_patch

    base = SimpleNamespace(
        lobby=SimpleNamespace(
            _game=SimpleNamespace(
                _patch=SimpleNamespace(templates={"BGS_104": SimpleNamespace(name=NOMI)})
            )
        )
    )
    _assert_minion_names_in_patch(base, {NOMI: 0.2})
    with pytest.raises(ValueError, match="not in this patch"):
        _assert_minion_names_in_patch(base, {"Nomi": 0.2})
