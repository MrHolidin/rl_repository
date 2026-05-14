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
