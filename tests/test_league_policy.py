import random

import pytest

from src.training.selfplay.league_policy import (
    OpponentKind,
    decide_opponent_kind,
    pfsp_sample,
    sample_scripted_key,
    self_play_enabled,
)


def test_self_play_start_episode():
    assert not self_play_enabled(episode=0, start_episode=10, has_self_play_config=True)
    assert self_play_enabled(episode=10, start_episode=10, has_self_play_config=True)


@pytest.mark.parametrize(
    "roll,expected",
    [
        (0.1, OpponentKind.CURRENT),
        (0.5, OpponentKind.FROZEN),
        (0.95, OpponentKind.SCRIPTED),
    ],
)
def test_decide_kind_with_frozen_pool(roll, expected):
    kind = decide_opponent_kind(
        roll,
        current_fraction=0.3,
        past_fraction=0.4,
        frozen_nonempty=True,
        has_current_agent=True,
    )
    assert kind == expected


def test_empty_frozen_pool_forces_current():
    kind = decide_opponent_kind(
        0.99,
        current_fraction=0.3,
        past_fraction=0.4,
        frozen_nonempty=False,
        has_current_agent=True,
    )
    assert kind == OpponentKind.CURRENT


def test_pfsp_prefers_stronger():
    rng = random.Random(0)
    counts = {0: 0, 1: 0}
    for _ in range(5000):
        sid = pfsp_sample([0, 1], [0.1, 0.9], rng)
        counts[sid] += 1
    assert counts[1] > counts[0] * 2


def test_sample_scripted_key_normalized():
    rng = random.Random(1)
    dist = {"a": 0.25, "b": 0.75}
    seen = {sample_scripted_key(dist, rng) for _ in range(200)}
    assert seen <= {"a", "b"}
