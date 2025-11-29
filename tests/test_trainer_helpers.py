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


def test_evaluate_agent_uses_series_helper_with_random_and_heuristic():
    agent = DummyAgent()

    series_outputs = [
        {"wins": 3, "draws": 1, "losses": 6},
        {"wins": 6, "draws": 2, "losses": 2},
        {"wins": 7, "draws": 0, "losses": 3},
    ]

    with patch.object(
        trainer_helpers, "play_series_vs_sampler", side_effect=series_outputs
    ) as mock_series, patch.object(
        trainer_helpers, "make_game", return_value=EvalTestEnv()
    ), patch.object(
        trainer_helpers, "RandomAgent", side_effect=lambda seed=None: DummyOpponent()
    ), patch.object(
        trainer_helpers, "HeuristicAgent", side_effect=lambda seed=None: DummyOpponent()
    ):
        metrics = evaluate_agent(
            agent,
            game_id="dummy",
            game_params={},
            num_episodes=10,
            deterministic=True,
            seed=123,
        )

    assert mock_series.call_count == 3
    random_metrics = {
        "eval_random/win_rate": 0.3,
        "eval_random/draw_rate": 0.1,
        "eval_random/loss_rate": 0.6,
    }
    heuristic_metrics = {
        "eval_heuristic/win_rate": 0.6,
        "eval_heuristic/draw_rate": 0.2,
        "eval_heuristic/loss_rate": 0.2,
    }
    smart_metrics = {
        "eval_smart_heuristic/win_rate": 0.7,
        "eval_smart_heuristic/draw_rate": 0.0,
        "eval_smart_heuristic/loss_rate": 0.3,
    }
    for key, value in {**random_metrics, **heuristic_metrics, **smart_metrics}.items():
        assert metrics[key] == value

