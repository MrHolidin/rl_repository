"""Tests for configuration schemas."""

from __future__ import annotations

from src.config import AppConfig


def test_app_config_parsing():
    data = {
        "game": {"id": "connect4", "params": {"rows": 6, "cols": 7}},
        "agent": {"id": "dqn", "params": {"learning_rate": 0.001}},
        "train": {
            "total_steps": 1000,
            "deterministic": False,
            "callbacks": [
                {"type": "eval", "params": {"interval": 100}},
                {"type": "checkpoint", "enabled": False},
            ],
        },
        "eval": {"enabled": True, "num_episodes": 5, "deterministic": True},
    }

    cfg = AppConfig.from_dict(data)
    assert cfg.game.id == "connect4"
    assert cfg.agent.id == "dqn"
    assert cfg.train.total_steps == 1000
    assert cfg.eval.enabled is True

