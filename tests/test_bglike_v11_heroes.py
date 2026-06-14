"""Tests for the hero-augmented obs (bglike_v5_heroes) and the v11_heroes net."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.agents.random_agent import RandomAgent
from src.bg_catalog.patch_context import load_patch_context
from src.envs.bglike.game import BGLikeGame
from src.envs.bglike.lobby_env import (
    BGLobbyEnv,
    OBS_KIND_BGLIKE_V5_HEROES,
    _obs_dim_for_kind,
)
from src.envs.bglike.obs import sorted_opponent_rows
from src.envs.bglike.obs_v5 import OBS_DIM_V5, build_observation_v5
from src.envs.bglike.obs_v5_heroes import (
    HERO_SELF_OFFSET,
    NUM_HERO_OBS_IDS,
    OBS_DIM_V5_HEROES,
    build_observation_v5_heroes,
    hero_obs_index,
)
from src.envs.bglike.seat_config import lobby_from_learned_seats
from src.models.bglike_structured_v11_heroes import BGLikeStructuredV11Heroes

PATCH_DIR = "data/bgcore/19_6_0_74257"


@pytest.fixture(scope="module")
def patch():
    return load_patch_context(PATCH_DIR)


def _hero_game(seed=3):
    g = BGLikeGame(seed=seed, with_heroes=True, patch_dir=PATCH_DIR)
    return g, g.initial_state()


# --------------------------------------------------------------------------- #
# Obs layout / content
# --------------------------------------------------------------------------- #


def test_hero_obs_dim_and_v5_prefix(patch):
    g, s = _hero_game()
    seat = s.current_player_index
    obs = build_observation_v5_heroes(s, seat, 0.0, is_my_turn=True, patch=patch)
    assert obs.shape == (OBS_DIM_V5_HEROES,)
    # The hero obs is a strict superset: its first OBS_DIM_V5 floats are exactly
    # the v5 obs (backward-compatible prefix).
    base = build_observation_v5(s, seat, 0.0, is_my_turn=True, patch=patch)
    assert np.array_equal(obs[:OBS_DIM_V5], base)


def test_self_hero_identity_and_effective_costs(patch):
    g, s = _hero_game()
    seat = s.current_player_index
    me = s.players[seat]
    me.hero = patch.heroes["millhouse"]  # buy/roll 2, upgrade +1
    obs = build_observation_v5_heroes(s, seat, 0.0, is_my_turn=True, patch=patch)

    idx = hero_obs_index(patch, me.hero)
    ident = obs[HERO_SELF_OFFSET : HERO_SELF_OFFSET + NUM_HERO_OBS_IDS]
    assert ident.argmax() == idx and ident.sum() == 1.0

    # Effective costs slice (just after the identity one-hot): roll/3, buy/5, level/max.
    c = HERO_SELF_OFFSET + NUM_HERO_OBS_IDS
    assert obs[c] == pytest.approx(2.0 / 3.0)          # Millhouse refresh = 2
    assert obs[c + 1] == pytest.approx(2.0 / 5.0)      # Millhouse buy = 2
    assert obs[c + 2] > 0.0                            # upgrade cost present


def test_no_hero_identity_is_zero_bucket(patch):
    g = BGLikeGame(seed=3, with_heroes=False, patch_dir=PATCH_DIR)
    s = g.initial_state()
    seat = s.current_player_index
    obs = build_observation_v5_heroes(s, seat, 0.0, is_my_turn=True, patch=patch)
    ident = obs[HERO_SELF_OFFSET : HERO_SELF_OFFSET + NUM_HERO_OBS_IDS]
    assert ident.argmax() == 0  # index 0 = none/unknown


def test_opponent_hero_onehots_aligned(patch):
    g, s = _hero_game(seed=5)
    seat = s.current_player_index
    obs = build_observation_v5_heroes(s, seat, 0.0, is_my_turn=True, patch=patch)

    from src.envs.bglike.obs_v5_heroes import HERO_OPP_OFFSET, HERO_SELF_DIM
    from src.envs.bglike.obs import MAX_OPPS

    opp_block = obs[HERO_OPP_OFFSET : HERO_OPP_OFFSET + MAX_OPPS * NUM_HERO_OBS_IDS]
    opp_block = opp_block.reshape(MAX_OPPS, NUM_HERO_OBS_IDS)
    rows = sorted_opponent_rows(s, seat)
    for j, row in enumerate(rows[:MAX_OPPS]):
        opp_seat = row[0]
        expected = hero_obs_index(patch, s.players[opp_seat].hero)
        assert opp_block[j].argmax() == expected and opp_block[j].sum() == 1.0


# --------------------------------------------------------------------------- #
# obs_kind plumbing
# --------------------------------------------------------------------------- #


def test_obs_kind_dim_and_env_builder():
    assert _obs_dim_for_kind(OBS_KIND_BGLIKE_V5_HEROES) == OBS_DIM_V5_HEROES
    cfgs = lobby_from_learned_seats((0,), agent_by_seat={0: RandomAgent(seed=1)}, seed=1)
    env = BGLobbyEnv(
        cfgs,
        learned_seats=(0,),
        patch_dir=PATCH_DIR,
        obs_kind=OBS_KIND_BGLIKE_V5_HEROES,
        with_heroes=True,
        seed=1,
    )
    env.reset(seed=1)
    obs = env.obs_for_seat(0)
    assert obs.shape == (OBS_DIM_V5_HEROES,)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def test_net_forward_and_value(patch):
    net = BGLikeStructuredV11Heroes(num_pool_indices=patch.num_pool_indices, num_identities=4)
    g, s = _hero_game()
    seat = s.current_player_index
    obs = build_observation_v5_heroes(s, seat, 0.0, is_my_turn=True, patch=patch)
    x = torch.from_numpy(np.stack([obs, obs])).float()
    state_emb, cache = net.encode_state(x)
    assert state_emb.shape == (2, net.state_dim)
    v = net.value_from_trunk(cache["trunk"])
    assert v.shape == (2,)
    # identity tail accepted too
    xi = torch.cat([x, torch.zeros(2, net.num_identities)], dim=1)
    assert net.encode_state(xi)[0].shape == (2, net.state_dim)


def test_net_full_policy_forward(patch):
    cfgs = lobby_from_learned_seats((0,), agent_by_seat={0: RandomAgent(seed=1)}, seed=1)
    env = BGLobbyEnv(
        cfgs,
        learned_seats=(0,),
        patch_dir=PATCH_DIR,
        obs_kind=OBS_KIND_BGLIKE_V5_HEROES,
        with_heroes=True,
        seed=1,
    )
    s = env.reset(seed=1)
    seat = s.current_player_index
    obs = env.obs_for_seat(seat)
    legal = env.legal_structured_actions_for_seat(seat)
    assert legal
    net = BGLikeStructuredV11Heroes(num_pool_indices=patch.num_pool_indices, num_identities=4)
    x = torch.from_numpy(obs[None, :]).float()
    logits, mask, values = net.policy_logits_and_value(x, [legal])
    assert logits.shape[0] == 1 and logits.shape[1] == len(legal)
    assert values.shape == (1,)
    assert torch.isfinite(logits[mask]).all()


def test_constructor_kwargs_roundtrip(patch):
    net = BGLikeStructuredV11Heroes(
        num_pool_indices=patch.num_pool_indices, num_identities=4, hero_hidden=48, hero_out=24
    )
    kw = net.get_constructor_kwargs()
    assert kw["hero_hidden"] == 48 and kw["hero_out"] == 24
    net2 = BGLikeStructuredV11Heroes(**kw)
    sd1, sd2 = net.state_dict(), net2.state_dict()
    assert sd1.keys() == sd2.keys()
    for k in sd1:
        assert sd1[k].shape == sd2[k].shape


def test_factory_builds_v11_heroes(patch):
    from src.models.ppo_policy_factory import (
        PPO_NETWORK_BGLIKE_STRUCTURED_V11_HEROES,
        build_ppo_actor_critic,
    )

    net = build_ppo_actor_critic(
        PPO_NETWORK_BGLIKE_STRUCTURED_V11_HEROES,
        observation_shape=(OBS_DIM_V5_HEROES,),
        num_actions=1,
        num_pool_indices=patch.num_pool_indices,
    )
    assert isinstance(net, BGLikeStructuredV11Heroes)


# --------------------------------------------------------------------------- #
# Backward compatibility
# --------------------------------------------------------------------------- #


def test_v5_obs_and_v11_unchanged(patch):
    # obs_v5 width is untouched; the plain v11 net still builds on it.
    assert OBS_DIM_V5 == 2536
    from src.models.bglike_structured_v11 import BGLikeStructuredV11

    net = BGLikeStructuredV11(num_pool_indices=patch.num_pool_indices, num_identities=4)
    g = BGLikeGame(seed=1, with_heroes=False, patch_dir=PATCH_DIR)
    s = g.initial_state()
    obs = build_observation_v5(s, s.current_player_index, 0.0, is_my_turn=True, patch=patch)
    assert obs.shape == (OBS_DIM_V5,)
    se, _ = net.encode_state(torch.from_numpy(obs[None, :]).float())
    assert se.shape == (1, net.state_dim)
