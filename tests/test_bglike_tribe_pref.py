"""Per-seat tribe-preference vector: draw, observation, purchase counter, shaping.

The vector is a hand-written stand-in for a DvD identity: drawn once per seat
per game, uniform in [-1, 1] per tribe, visible in that seat's observation, and
paid out as ``coef * v[tribe]`` for every minion of that tribe the seat buys.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import src.envs  # noqa: F401
from src.bg_catalog.cards import Race
from src.envs.bglike.lobby_env import BGLobbyEnv, OBS_KIND_BGLIKE_V7_PREF
from src.envs.bglike.obs_v7_pref import (
    OBS_DIM_V7_PREF,
    TRIBE_PREF_DIM,
    TRIBE_PREF_OFFSET_V7,
)
from src.envs.bglike.seat_config import lobby_from_learned_seats
from src.envs.bglike.tribe_pref import (
    NUM_TRIBES,
    TRIBES,
    draw_tribe_pref,
    pref_reward_for_counts,
    pref_value,
)
from src.agents.random_agent import RandomAgent
from src.training.bglike_perspective import BGLikeAgentPerspectiveEnv

PATCH = "data/bgcore/19_6_0_74257"


def _env(seed=5, with_pref=True):
    seats = tuple(range(8))
    env = BGLobbyEnv(
        lobby_from_learned_seats(seats, agent_by_seat={s: RandomAgent(seed=s) for s in seats}),
        learned_seats=seats,
        training_seats=seats,
        seed=seed,
        patch_dir=PATCH,
        obs_kind=OBS_KIND_BGLIKE_V7_PREF,
        with_heroes=True,
        with_tribe_pref=with_pref,
    )
    env.reset(seed=seed)
    return env


# --------------------------------------------------------------------------- #
# The vector itself
# --------------------------------------------------------------------------- #


def test_draw_is_one_component_per_tribe_in_range():
    rng = np.random.default_rng(0)
    v = draw_tribe_pref(rng)
    assert len(v) == NUM_TRIBES == 7
    assert all(-1.0 <= x <= 1.0 for x in v)
    assert Race.ALL not in TRIBES  # ALL is every tribe, not a tribe of its own


def test_pref_value_scores_all_as_the_mean_and_tribeless_as_zero():
    v = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert pref_value(v, Race.BEAST) == pytest.approx(1.0)
    assert pref_value(v, Race.DEMON) == pytest.approx(0.0)
    assert pref_value(v, None) == pytest.approx(0.0)
    assert pref_value(v, Race.ALL) == pytest.approx(1.0 / 7)
    # No vector configured reads as no preference, never as an error.
    assert pref_value((), Race.BEAST) == 0.0


def test_reward_for_counts_is_linear_in_the_counts():
    v = tuple(float(i) for i in range(7))
    counts = {Race.BEAST: 2, Race.DEMON: 3, None: 5}
    assert pref_reward_for_counts(v, counts) == pytest.approx(2 * 0.0 + 3 * 1.0)


# --------------------------------------------------------------------------- #
# Draw + observation
# --------------------------------------------------------------------------- #


def test_every_seat_draws_a_vector_including_the_opponent_seats():
    """Frozen/scripted seats are drawn for too — the game fills all eight."""
    st = _env().state
    prefs = [tuple(st.players[s].tribe_pref) for s in range(8)]
    assert all(len(p) == TRIBE_PREF_DIM for p in prefs)
    assert len(set(prefs)) == 8  # independent draws, not one shared vector


def test_observation_carries_the_seats_own_vector():
    env = _env()
    st = env.state
    for seat in (0, 3, 7):
        obs = env.obs_for_seat(seat)
        assert obs.shape == (OBS_DIM_V7_PREF,)
        assert np.allclose(obs[TRIBE_PREF_OFFSET_V7:], st.players[seat].tribe_pref, atol=1e-6)


def test_a_seat_cannot_read_another_seats_vector():
    env = _env()
    st = env.state
    tail0 = env.obs_for_seat(0)[TRIBE_PREF_OFFSET_V7:]
    assert not np.allclose(tail0, st.players[1].tribe_pref, atol=1e-6)


def test_disabled_leaves_the_block_at_zero():
    env = _env(with_pref=False)
    assert env.state.players[0].tribe_pref == ()
    assert np.allclose(env.obs_for_seat(0)[TRIBE_PREF_OFFSET_V7:], 0.0)


def test_same_seed_reproduces_the_vectors():
    a = [tuple(_env(seed=9).state.players[s].tribe_pref) for s in range(8)]
    b = [tuple(_env(seed=9).state.players[s].tribe_pref) for s in range(8)]
    assert a == b


# --------------------------------------------------------------------------- #
# Purchase counter (engine)
# --------------------------------------------------------------------------- #


def test_purchases_are_counted_by_tribe():
    env = _env(seed=11)
    env.drain_until_lobby_done(deterministic=True)
    st = env.state
    totals = {s: sum(st.players[s].bought_tribe_counts.values()) for s in range(8)}
    assert sum(totals.values()) > 0
    for seat, n in totals.items():
        counts = st.players[seat].bought_tribe_counts
        assert all(k is None or isinstance(k, Race) for k in counts)
        assert n == sum(counts.values())


# --------------------------------------------------------------------------- #
# Shaping term
# --------------------------------------------------------------------------- #


class _Seat:
    def __init__(self, pref):
        self.tribe_pref = pref
        self.bought_tribe_counts = {}


def _wrapper(coef=0.02, prefs=None):
    env = BGLikeAgentPerspectiveEnv.__new__(BGLikeAgentPerspectiveEnv)
    env._tribe_pref_coef = coef
    env._tribe_pref_seen = {}
    prefs = prefs or [(1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0)] * 2
    env._bg_base = SimpleNamespace(
        state=SimpleNamespace(players=[_Seat(p) for p in prefs])
    )
    return env


def test_pays_on_the_purchase_delta_once():
    env = _wrapper()
    p = env._bg_base.state.players[0]
    p.bought_tribe_counts[Race.BEAST] = 1
    assert env._tribe_pref_reward({"acting_seat": 0}) == pytest.approx(0.02)
    # Nothing new bought → nothing paid again.
    assert env._tribe_pref_reward({"acting_seat": 0}) == 0.0
    p.bought_tribe_counts[Race.BEAST] = 3
    assert env._tribe_pref_reward({"acting_seat": 0}) == pytest.approx(0.04)


def test_a_negative_component_is_a_penalty():
    env = _wrapper()
    env._bg_base.state.players[0].bought_tribe_counts[Race.DEMON] = 2
    assert env._tribe_pref_reward({"acting_seat": 0}) == pytest.approx(-0.04)


def test_credit_is_per_seat():
    env = _wrapper(prefs=[(1.0, 0, 0, 0, 0, 0, 0), (-1.0, 0, 0, 0, 0, 0, 0)])
    env._bg_base.state.players[0].bought_tribe_counts[Race.BEAST] = 1
    env._bg_base.state.players[1].bought_tribe_counts[Race.BEAST] = 1
    assert env._tribe_pref_reward({"acting_seat": 0}) == pytest.approx(0.02)
    assert env._tribe_pref_reward({"acting_seat": 1}) == pytest.approx(-0.02)


def test_disabled_by_default():
    env = _wrapper(coef=0.0)
    env._bg_base.state.players[0].bought_tribe_counts[Race.BEAST] = 5
    assert env._tribe_pref_reward({"acting_seat": 0}) == 0.0


# --------------------------------------------------------------------------- #
# The net actually reads it
# --------------------------------------------------------------------------- #


def _net():
    from src.bg_catalog.patch_context import load_patch_context
    from src.models.bglike_structured_v13 import BGLikeStructuredV13

    ctx = load_patch_context(PATCH)
    return BGLikeStructuredV13(
        num_pool_indices=ctx.num_pool_indices,
        slot_hidden=32,
        entity_attention_layers=1,
        card_patch_dir=PATCH,
    ).eval()


def test_changing_the_vector_changes_the_trunk():
    """The whole arm is inert if the net can ignore the vector."""
    net = _net()
    torch.manual_seed(0)
    x = torch.randn(4, OBS_DIM_V7_PREF)
    y = x.clone()
    y[:, TRIBE_PREF_OFFSET_V7:] = -x[:, TRIBE_PREF_OFFSET_V7:]
    _, ca = net.encode_state(x)
    _, cb = net.encode_state(y)
    assert not torch.allclose(ca["trunk"], cb["trunk"], atol=1e-6)


def test_the_vector_reaches_the_critic():
    net = _net()
    # An untrained critic head is zero-initialized, so give it weights before
    # asking whether the vector reaches it.
    torch.nn.init.normal_(net.critic_dist[-1].weight, std=0.05)
    torch.manual_seed(0)
    x = torch.randn(4, OBS_DIM_V7_PREF)
    y = x.clone()
    y[:, TRIBE_PREF_OFFSET_V7:] = -x[:, TRIBE_PREF_OFFSET_V7:]
    va = net.value_from_trunk(net.encode_state(x)[1]["trunk"])
    vb = net.value_from_trunk(net.encode_state(y)[1]["trunk"])
    assert not torch.allclose(va, vb, atol=1e-6)


def test_num_identities_is_pinned_to_the_tribe_count():
    from src.bg_catalog.patch_context import load_patch_context
    from src.models.bglike_structured_v13 import BGLikeStructuredV13

    ctx = load_patch_context(PATCH)
    with pytest.raises(ValueError, match="pins num_identities"):
        BGLikeStructuredV13(
            num_pool_indices=ctx.num_pool_indices,
            num_identities=4,
            card_patch_dir=PATCH,
        )
