import pytest

from src.agents.base_agent import BaseAgent
from src.training.opponent_sampler import MiniBGMixedOpponentSampler


def test_minibg_mixed_rejects_random_in_bots():
    with pytest.raises(ValueError, match="random_fraction"):
        MiniBGMixedOpponentSampler(seed=0, random_fraction=0.5, bots=["tempo", "random"])


def test_equal_opponent_mass_sets_random_share():
    s = MiniBGMixedOpponentSampler(
        seed=0,
        bots=["tempo", "buffer_t2"],
        equal_opponent_mass=True,
    )
    assert abs(s.random_fraction - 1.0 / 3.0) < 1e-9


def test_minibg_mixed_samples_both_kinds():
    s = MiniBGMixedOpponentSampler(
        seed=42,
        random_fraction=0.5,
        bots=["tempo", "buffer_t2"],
    )
    kinds = []
    for ep in range(400):
        s.prepare(ep)
        a = s.sample()
        kinds.append(type(a).__name__)
    assert "RandomAgent" in kinds
    assert "MiniBGHeuristicAgent" in kinds


def test_minibg_mixed_unknown_bot():
    with pytest.raises(ValueError, match="unknown bot"):
        MiniBGMixedOpponentSampler(seed=0, random_fraction=0.5, bots=["not_a_bot"])


class _StubLearner(BaseAgent):
    def act(self, obs, legal_mask=None, deterministic=False):
        return 0

    def save(self, path: str) -> None:
        pass

    @classmethod
    def load(cls, path: str, **kwargs):
        raise NotImplementedError


def test_minibg_mixed_self_play_requires_agent():
    with pytest.raises(ValueError, match="learning agent"):
        MiniBGMixedOpponentSampler(
            seed=0,
            random_fraction=0.5,
            bots=["tempo"],
            learning_agent=None,
            self_play={"current_self_fraction": 0.1},
        )


def test_minibg_mixed_self_play_wraps_learner():
    from src.training.selfplay.opponent_pool import SelfPlayOpponent

    learner = _StubLearner()
    s = MiniBGMixedOpponentSampler(
        seed=0,
        random_fraction=0.0,
        bots=["tempo"],
        learning_agent=learner,
        self_play={"current_self_fraction": 1.0, "start_episode": 0},
    )
    s.prepare(0)
    opp = s.sample()
    assert isinstance(opp, SelfPlayOpponent)
