"""Nomi shaping: pay the acting seat per point of tavern elemental buff it adds."""

from types import SimpleNamespace

from src.training.bglike_perspective import BGLikeAgentPerspectiveEnv


class _Seat:
    def __init__(self, bonus=0):
        self.shop_elemental_bonus = bonus


def _env(coef=0.02, cap=10, seats=2):
    """An instance without __init__ — only the shaping fields are exercised."""
    env = BGLikeAgentPerspectiveEnv.__new__(BGLikeAgentPerspectiveEnv)
    env._elem_bonus_coef = coef
    env._elem_bonus_cap = cap
    env._elem_bonus_seen = {}
    env._elem_bonus_paid = {}
    env._bg_base = SimpleNamespace(state=SimpleNamespace(players=[_Seat() for _ in range(seats)]))
    return env


def _bump(env, seat, by):
    env._bg_base.state.players[seat].shop_elemental_bonus += by
    return env._elemental_bonus_reward({"acting_seat": seat})


def test_pays_per_point_gained():
    env = _env()
    assert _bump(env, 0, 1) == 0.02
    assert _bump(env, 0, 2) == 0.04


def test_pays_nothing_when_the_counter_does_not_move():
    env = _env()
    _bump(env, 0, 1)
    assert env._elemental_bonus_reward({"acting_seat": 0}) == 0.0


def test_credit_stops_at_the_cap():
    env = _env(cap=10)
    assert _bump(env, 0, 8) == 0.02 * 8
    # only the two points below the cap are creditable, the rest is free
    assert _bump(env, 0, 5) == 0.02 * 2
    assert _bump(env, 0, 5) == 0.0
    assert env._elem_bonus_paid[0] == 10


def test_cap_and_ledger_are_per_seat():
    env = _env(cap=3, seats=2)
    assert _bump(env, 0, 3) == 0.02 * 3
    assert _bump(env, 0, 1) == 0.0
    assert _bump(env, 1, 2) == 0.02 * 2


def test_disabled_by_default_coefficient():
    env = _env(coef=0.0)
    assert _bump(env, 0, 5) == 0.0


def test_no_acting_seat_pays_nothing():
    env = _env()
    env._bg_base.state.players[0].shop_elemental_bonus = 4
    assert env._elemental_bonus_reward({}) == 0.0


def test_full_cap_is_twenty_percent_of_the_best_placement_reward():
    """The coefficient is chosen against the terminal reward, (9-2*place)/7."""
    from src.envs.bglike.placement import placement_reward

    env = _env(coef=0.02, cap=10)
    total = _bump(env, 0, 10)
    assert total == 0.2
    assert total == 0.2 * placement_reward(1)
