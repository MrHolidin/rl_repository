"""Per-seat independent opponent assignment in BGLike training lobbies."""

from __future__ import annotations

from src.agents.random_agent import RandomAgent
from src.envs.bglike.actions import NUM_PLAYERS
from src.envs.bglike.seat_config import build_training_lobby_configs


def test_build_training_lobby_configs_distinct_opponents_per_seat():
    learner = RandomAgent(seed=1)
    opponents = {s: RandomAgent(seed=10 + s) for s in range(1, NUM_PLAYERS)}
    configs = build_training_lobby_configs((0,), learner, opponents)
    assert len(configs) == NUM_PLAYERS
    assert configs[0].agent is learner
    for seat in range(1, NUM_PLAYERS):
        assert configs[seat].agent is opponents[seat]
    assert len({id(configs[s].agent) for s in range(1, NUM_PLAYERS)}) == NUM_PLAYERS - 1


def test_make_bglike_reset_samples_opponents_per_seat():
    from src.training.bglike_perspective import make_bglike_agent_perspective_env
    from src.training.opponent_sampler import RandomOpponentSampler

    class _CountingSampler(RandomOpponentSampler):
        def __init__(self) -> None:
            super().__init__(seed=42)
            self.sample_calls = 0

        def sample(self):
            self.sample_calls += 1
            return super().sample()

    sampler = _CountingSampler()
    env = make_bglike_agent_perspective_env(
        sampler,
        num_current_seats=4,
        seed=7,
        patch_dir="data/bgcore/15_6_2_36393",
    )
    env.set_learner_agent(RandomAgent(seed=1))
    env.reset()
    assert sampler.sample_calls == NUM_PLAYERS - 4
