import os
import tempfile

import pytest

from src.agents import RandomAgent
from src.agents.base_agent import BaseAgent
from src.envs.minibg.heuristic_bots.agent_adapter import MiniBGHeuristicAgent
from src.training.selfplay.opponent_pool import (
    OpponentPool,
    ScriptedOpponentsSpec,
    SelfPlayConfig,
    SelfPlayOpponent,
)


def test_minibg_pool_rejects_unknown_bot():
    with pytest.raises(ValueError, match="unknown"):
        OpponentPool(
            device="cpu",
            seed=0,
            self_play_config=None,
            scripted=ScriptedOpponentsSpec("minibg", {"not_a_bot": 1.0}),
        )


def test_minibg_pool_samples_random_and_scripted():
    pool = OpponentPool(
        device="cpu",
        seed=42,
        self_play_config=None,
        scripted=ScriptedOpponentsSpec(
            "minibg",
            {"random": 0.5, "t1_random": 0.25, "t_up_random": 0.25},
        ),
    )
    kinds = []
    for ep in range(400):
        a = pool.sample_opponent(ep)
        kinds.append(type(a).__name__)
    assert "RandomAgent" in kinds
    assert "MiniBGHeuristicAgent" in kinds


def test_minibg_equal_mass_fraction_in_spec():
    d = {"random": 1.0 / 3.0, "t1_random": 1.0 / 3.0, "t_up_random": 1.0 / 3.0}
    pool = OpponentPool(
        device="cpu",
        seed=0,
        self_play_config=None,
        scripted=ScriptedOpponentsSpec("minibg", d),
    )
    assert abs(sum(pool._scripted.distribution.values()) - 1.0) < 1e-9


class _StubLearner(BaseAgent):
    def act(self, obs, legal_mask=None, deterministic=False):
        return 0

    def save(self, path: str) -> None:
        pass

    @classmethod
    def load(cls, path: str, **kwargs):
        raise NotImplementedError


def test_minibg_pool_self_play_wraps_learner():
    learner = _StubLearner()
    cfg = SelfPlayConfig(
        start_episode=0,
        current_self_fraction=1.0,
        past_self_fraction=0.0,
        max_frozen_agents=4,
        save_every=100,
        frozen_ema_beta=0.05,
    )
    pool = OpponentPool(
        device="cpu",
        seed=0,
        self_play_config=cfg,
        scripted=ScriptedOpponentsSpec("minibg", {"t1_random": 1.0}),
        current_agent=learner,
    )
    opp = pool.sample_opponent(0)
    assert isinstance(opp, SelfPlayOpponent)


def test_frozen_pool_evicts_lowest_winrate_and_oldest_on_tie():
    """max_frozen_agents caps pool; remove min EMA win rate, then lowest episode when tied."""
    cfg = SelfPlayConfig(
        start_episode=0,
        current_self_fraction=0.0,
        past_self_fraction=1.0,
        max_frozen_agents=2,
        save_every=100,
        frozen_ema_beta=0.05,
    )
    pool = OpponentPool(
        device="cpu",
        seed=0,
        self_play_config=cfg,
        scripted=ScriptedOpponentsSpec("minibg", {"t1_random": 1.0}),
    )
    paths = []
    try:
        for _ in range(3):
            fd, p = tempfile.mkstemp(suffix=".pt")
            os.close(fd)
            paths.append(p)
        pool.add_frozen_agent(paths[0], episode=100)
        pool.add_frozen_agent(paths[1], episode=200)
        pool.add_frozen_agent(paths[2], episode=300)
        assert len(pool.frozen_agents) == 2
        eps = {fa.episode for fa in pool.frozen_agents}
        assert eps == {200, 300}
    finally:
        for p in paths:
            try:
                os.unlink(p)
            except OSError:
                pass


def test_frozen_pool_evicts_lower_ema_when_winrates_differ():
    cfg = SelfPlayConfig(
        start_episode=0,
        current_self_fraction=0.0,
        past_self_fraction=1.0,
        max_frozen_agents=2,
        save_every=100,
        frozen_ema_beta=0.05,
    )
    pool = OpponentPool(
        device="cpu",
        seed=0,
        self_play_config=cfg,
        scripted=ScriptedOpponentsSpec("minibg", {"t1_random": 1.0}),
    )
    paths = []
    try:
        for _ in range(3):
            fd, p = tempfile.mkstemp(suffix=".pt")
            os.close(fd)
            paths.append(p)
        pool.add_frozen_agent(paths[0], episode=10)
        pool.add_frozen_agent(paths[1], episode=20)
        pool.frozen_agents[0].ema_win_rate = 0.9
        pool.frozen_agents[1].ema_win_rate = 0.1
        pool.add_frozen_agent(paths[2], episode=30)
        assert len(pool.frozen_agents) == 2
        eps = {fa.episode for fa in pool.frozen_agents}
        assert eps == {10, 30}
    finally:
        for p in paths:
            try:
                os.unlink(p)
            except OSError:
                pass


def test_checkpoint_skips_frozen_when_no_self_play_config():
    from src.training.opponent_sampler import OpponentPoolSampler

    pool = OpponentPool(
        device="cpu",
        seed=0,
        self_play_config=None,
        scripted=ScriptedOpponentsSpec("minibg", {"t1_random": 1.0}),
    )
    sampler = OpponentPoolSampler(opponent_pool=pool)
    sampler.on_checkpoint("/nonexistent/foo.pt", 0)
    assert pool.frozen_agents == []
