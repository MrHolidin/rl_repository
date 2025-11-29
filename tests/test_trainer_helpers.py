from unittest.mock import patch

import numpy as np

from notebooks.utils import trainer_helpers
from notebooks.utils.trainer_helpers import evaluate_agent
from src.envs.base import StepResult, TurnBasedEnv


class DummyAgent:
    def act(self, obs, legal_mask=None, deterministic=False):
        return 0


class DummyOpponent:
    def act(self, obs, legal_mask=None, deterministic=False):
        return 0


class EvalTestEnv(TurnBasedEnv):
    """Environment that always reports winner=1 regardless of moves."""

    def __init__(self):
        self._legal_mask = np.ones(1, dtype=bool)

    def reset(self, seed=None):
        return np.zeros(1, dtype=np.float32)

    def step(self, action: int) -> StepResult:
        return StepResult(
            obs=self.reset(),
            reward=0.0,
            terminated=True,
            truncated=False,
            info={"winner": 1},
        )

    @property
    def legal_actions_mask(self) -> np.ndarray:
        return self._legal_mask

    def current_player(self) -> int:
        return 0

    def render(self) -> None:
        pass


def test_evaluate_agent_alternates_start_positions():
    agent = DummyAgent()

    with patch.object(trainer_helpers, "make_game", return_value=EvalTestEnv()), patch.object(
        trainer_helpers, "RandomAgent", side_effect=lambda seed=None: DummyOpponent()
    ):
        metrics = evaluate_agent(
            agent,
            game_id="dummy",
            game_params={},
            num_episodes=4,
            deterministic=True,
            seed=123,
        )

    assert metrics["eval/win_rate"] == 0.5
    assert metrics["eval/loss_rate"] == 0.5

