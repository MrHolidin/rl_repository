"""End-to-end-ish checks for v11_heroes training wiring:

* the shipped config resolves to the hero obs (2683) and a hero net;
* heroes are simulated on every seat across a full game;
* a real collect + PPO update runs and gradients reach the hero modules;
* the with_heroes <-> hero-net consistency guard fires both ways.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

from src.bg_catalog.patch_context import load_patch_context
from src.envs.bglike.action_map import NUM_ENV_ACTIONS
from src.envs.bglike.obs_v5_heroes import (
    HERO_SELF_OFFSET,
    NUM_HERO_OBS_IDS,
    OBS_DIM_V5_HEROES,
)
from src.registry import make_agent
from src.training.bg_network_policy import validate_heroes_consistency

PATCH_DIR = "data/bgcore/19_6_0_74257"
CONFIG = "configs/bglike/ppo_v11_heroes_74257.yaml"


@pytest.fixture(scope="module")
def patch():
    return load_patch_context(PATCH_DIR)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def test_config_resolves_to_hero_obs():
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    gp = cfg["game"]["params"]
    ap = cfg["agent"]["params"]
    assert ap["network_type"] == "bglike_structured_v11_heroes"
    assert gp["obs_kind"] == "bglike_v5_heroes"
    assert gp["with_heroes"] is True

    from src.training.obs_sizing import apply_bg_observation_defaults

    agent_params: dict = {}
    apply_bg_observation_defaults("bglike", agent_params, obs_kind=gp["obs_kind"])
    assert agent_params["observation_shape"] == (OBS_DIM_V5_HEROES,)

    # The guard is happy with the shipped (coherent) config.
    validate_heroes_consistency("bglike", ap["network_type"], gp)


# --------------------------------------------------------------------------- #
# Consistency guard
# --------------------------------------------------------------------------- #


def test_guard_with_heroes_on_non_hero_net_raises():
    with pytest.raises(ValueError, match="cannot observe"):
        validate_heroes_consistency(
            "bglike", "bglike_structured_v11", {"with_heroes": True}
        )


def test_guard_hero_net_without_heroes_raises():
    with pytest.raises(ValueError, match="with_heroes"):
        validate_heroes_consistency(
            "bglike", "bglike_structured_v11_heroes", {"with_heroes": False}
        )


def test_guard_allows_coherent_combos():
    # hero net + heroes, and non-hero net + no heroes: both fine.
    validate_heroes_consistency(
        "bglike", "bglike_structured_v11_heroes", {"with_heroes": True}
    )
    validate_heroes_consistency("bglike", "bglike_structured_v11", {"with_heroes": False})
    validate_heroes_consistency("bglike", "bglike_structured_v8", {})


# --------------------------------------------------------------------------- #
# Heroes simulated everywhere
# --------------------------------------------------------------------------- #


def test_heroes_on_every_seat_across_a_game(patch):
    from src.envs.bglike.game import BGLikeGame
    from src.envs.bglike.obs_v5_heroes import build_observation_v5_heroes

    g = BGLikeGame(seed=4, with_heroes=True, patch_dir=PATCH_DIR)
    s = g.initial_state()
    rng = np.random.default_rng(4)
    seen_rounds = set()
    steps = 0
    while not g.is_terminal(s) and steps < 1500:
        # Every alive seat always has a hero assigned.
        for seat in s.alive:
            assert s.players[seat].hero is not None
        seat = s.current_player_index
        # The acting seat's hero obs block carries a real identity (non-zero bucket).
        obs = build_observation_v5_heroes(s, seat, 0.0, is_my_turn=True, patch=patch)
        ident = obs[HERO_SELF_OFFSET : HERO_SELF_OFFSET + NUM_HERO_OBS_IDS]
        assert ident.sum() == 1.0 and ident.argmax() >= 1
        seen_rounds.add(s.round_number)
        legal = g.legal_actions(s)
        if not legal:
            break
        s = g.apply_action(s, int(rng.choice(legal)))
        steps += 1
    assert len(seen_rounds) >= 3  # exercised several rounds (combat transitions)


# --------------------------------------------------------------------------- #
# Real collect + PPO update → gradients reach the hero modules
# --------------------------------------------------------------------------- #


def _build_hero_agent(patch, rollout_steps):
    return make_agent(
        "ppo",
        network_type="bglike_structured_v11_heroes",
        observation_shape=(OBS_DIM_V5_HEROES,),
        observation_type="vector",
        num_actions=int(NUM_ENV_ACTIONS),
        num_pool_indices=patch.num_pool_indices,
        num_identities=4,
        slot_hidden_channels=32,
        card_emb_dim=16,
        entity_attention_layers=2,
        rollout_steps=rollout_steps,
        ppo_epochs=1,
        minibatch_size=32,
        learning_rate=1e-3,
        value_coef=0.05,
        device="cpu",
    )


def test_real_collect_and_update_grad_reaches_heroes(patch):
    from src.agents.ppo_structured_minibg_agent import (
        INFO_STRUCT_LEGAL,
        INFO_STRUCT_NEXT_LEGAL,
    )
    from src.training.bglike_perspective import (
        apply_bglike_segment_closures_after_observe,
        make_bglike_agent_perspective_env,
    )
    from src.training.opponent_sampler import RandomOpponentSampler
    from src.training.trainer import Transition

    torch.manual_seed(0)
    rollout_steps = 48
    agent = _build_hero_agent(patch, rollout_steps)
    agent.train()

    env = make_bglike_agent_perspective_env(
        RandomOpponentSampler(seed=1),
        current_seats=(0,),
        seed=1,
        patch_dir=PATCH_DIR,
        obs_kind="bglike_v5_heroes",
        with_heroes=True,
    )
    if hasattr(env, "set_learner_agent"):
        env.set_learner_agent(agent)

    # Real collection loop (mirrors distributed_trainer._collect_until_steps_structured).
    safety = 0
    while len(agent.rollout_buffer) < rollout_steps and safety < 50:
        safety += 1
        obs = env.reset()
        last_info: dict = {}
        while not env.done:
            legal = env.legal_structured_actions()
            if not legal:
                break
            struct_act, board_perm, idx = agent.act_structured(
                obs, legal, env, deterministic=False
            )
            step = env.step_structured(struct_act, board_perm=board_perm)
            info = step.info if isinstance(step.info, dict) else {}
            next_sl = [] if env.done else list(env.legal_structured_actions())
            agent.observe(
                Transition(
                    obs=obs,
                    action=idx,
                    reward=float(step.reward),
                    next_obs=step.obs,
                    terminated=step.terminated,
                    truncated=step.truncated,
                    info={
                        **info,
                        INFO_STRUCT_LEGAL: legal,
                        INFO_STRUCT_NEXT_LEGAL: next_sl,
                    },
                    legal_mask=None,
                    next_legal_mask=None,
                )
            )
            apply_bglike_segment_closures_after_observe(env, info)
            obs = step.obs
            if env.done:
                last_info = info
                break
        env.notify_episode_end(last_info)

    assert len(agent.rollout_buffer) >= rollout_steps, "failed to collect a rollout"

    net = agent.policy_net
    before = net.hero_encoder[0].weight.detach().clone()

    metrics = agent.update()

    # A real PPO step ran and produced finite losses.
    assert metrics, "update returned no metrics (buffer under capacity?)"
    for k, v in metrics.items():
        if isinstance(v, float):
            assert np.isfinite(v), f"non-finite metric {k}={v}"

    # Gradient actually reached the hero scalar encoder (weights moved).
    after = net.hero_encoder[0].weight.detach()
    assert not torch.allclose(before, after), "hero_encoder did not update — no gradient"
    # And the widened opponent projection (opponent-hero one-hot path) has a grad.
    assert net.opp_proj.weight.grad is not None
    assert torch.isfinite(net.opp_proj.weight.grad).all()


def _collect_rollout(agent, env, rollout_steps):
    """Real collection loop (mirrors distributed_trainer._collect_until_steps_structured)."""
    from src.agents.ppo_structured_minibg_agent import (
        INFO_STRUCT_LEGAL,
        INFO_STRUCT_NEXT_LEGAL,
    )
    from src.training.bglike_perspective import (
        apply_bglike_segment_closures_after_observe,
    )
    from src.training.trainer import Transition

    safety = 0
    while len(agent.rollout_buffer) < rollout_steps and safety < 50:
        safety += 1
        obs = env.reset()
        last_info: dict = {}
        while not env.done:
            legal = env.legal_structured_actions()
            if not legal:
                break
            struct_act, board_perm, idx = agent.act_structured(
                obs, legal, env, deterministic=False
            )
            step = env.step_structured(struct_act, board_perm=board_perm)
            info = step.info if isinstance(step.info, dict) else {}
            next_sl = [] if env.done else list(env.legal_structured_actions())
            agent.observe(
                Transition(
                    obs=obs,
                    action=idx,
                    reward=float(step.reward),
                    next_obs=step.obs,
                    terminated=step.terminated,
                    truncated=step.truncated,
                    info={**info, INFO_STRUCT_LEGAL: legal, INFO_STRUCT_NEXT_LEGAL: next_sl},
                    legal_mask=None,
                    next_legal_mask=None,
                )
            )
            apply_bglike_segment_closures_after_observe(env, info)
            obs = step.obs
            if env.done:
                last_info = info
                break
        env.notify_episode_end(last_info)


def test_rnd_real_collect_and_update(patch, monkeypatch):
    """The RND branch of the PPO update runs on a REAL collected rollout: the
    intrinsic stream is built, combined, and its losses reach the new heads."""
    from src.training.bglike_perspective import make_bglike_agent_perspective_env
    from src.training.opponent_sampler import RandomOpponentSampler

    # grad-norm decomposition is opt-in (incompatible with the compiled host
    # update); this test is single-process / no compile, so enable it to exercise
    # the rnd/gradnorm_* path.
    monkeypatch.setenv("RL_RND_GRADNORM", "1")
    torch.manual_seed(0)
    rollout_steps = 64
    agent = make_agent(
        "ppo",
        network_type="bglike_structured_v11_heroes",
        observation_shape=(OBS_DIM_V5_HEROES,),
        observation_type="vector",
        num_actions=int(NUM_ENV_ACTIONS),
        num_pool_indices=patch.num_pool_indices,
        num_identities=4,
        slot_hidden_channels=32,
        card_emb_dim=16,
        entity_attention_layers=2,
        rollout_steps=rollout_steps,
        ppo_epochs=1,
        minibatch_size=32,
        learning_rate=1e-3,
        value_coef=0.05,
        device="cpu",
        rnd={
            "enabled": True,
            "warmup_rounds": 0,  # exercise the bonus on the very first update
            "int_coef": 1.0,
            "value_coef_int": 0.5,
            "embed_dim": 32,
            "target_hidden": 32,
            "predictor_hidden": 32,
            "predictor_layers": 1,
        },
    )
    agent.train()
    assert agent._rnd_enabled and agent.policy_net.with_value_int is True

    env = make_bglike_agent_perspective_env(
        RandomOpponentSampler(seed=1),
        current_seats=(0,),
        seed=1,
        patch_dir=PATCH_DIR,
        obs_kind="bglike_v5_heroes",
        with_heroes=True,
    )
    if hasattr(env, "set_learner_agent"):
        env.set_learner_agent(agent)

    _collect_rollout(agent, env, rollout_steps)
    assert len(agent.rollout_buffer) >= rollout_steps, "failed to collect a rollout"

    before_vi = agent.policy_net.value_int[0].weight.detach().clone()
    metrics = agent.update()

    assert metrics, "update returned no metrics"
    for k, v in metrics.items():
        if isinstance(v, float):
            assert np.isfinite(v), f"non-finite metric {k}={v}"
    # A real rollout closes at least one turn → the intrinsic stream had nodes.
    assert metrics.get("rnd/num_nodes", 0) >= 1
    assert "rnd/value_loss" in metrics and "rnd/adv_int_std" in metrics
    # Intrinsic value head actually trained (gradient reached it).
    after_vi = agent.policy_net.value_int[0].weight.detach()
    assert not torch.allclose(before_vi, after_vi), "value_int head did not update"
    # The obs/return normalization stats advanced this update.
    assert float(agent.rnd.obs_count.item()) > 1.0

    # Loss-decomposition + grad-balance metrics are present...
    for key in (
        "loss/policy", "loss/value", "loss/entropy", "loss/value_int", "loss/predictor",
        "rnd/gradnorm_policy", "rnd/gradnorm_value", "rnd/gradnorm_value_int",
    ):
        assert key in metrics, f"missing metric {key}"
    # ...and the weighted additive members reconstruct the reported total loss
    # (no battle head here, so those are the only terms).
    recon = (
        metrics["loss/policy"]
        + metrics["loss/value"]
        - metrics["loss/entropy"]
        + metrics["loss/value_int"]
        + metrics["loss/predictor"]
    )
    assert np.isclose(metrics["loss"], recon, atol=1e-4), (metrics["loss"], recon)
    # value_int actually pulls on the shared trunk (non-zero grad contribution).
    assert metrics["rnd/gradnorm_value_int"] > 0.0
