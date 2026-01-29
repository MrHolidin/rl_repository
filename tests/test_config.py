"""Tests for configuration schemas."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.config import AppConfig, load_config


def test_app_config_parsing():
    data = {
        "game": {"id": "connect4", "params": {"rows": 6, "cols": 7}},
        "agent": {"id": "dqn", "params": {"learning_rate": 0.001}},
        "train": {
            "total_steps": 1000,
            "deterministic": False,
            "opponent_sampler": {"type": "random", "params": {}},
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
    assert cfg.seed is None

    chk = next(cb for cb in cfg.train.callbacks if cb.type == "checkpoint")
    assert chk.enabled is False


def test_app_config_seed():
    data = {
        "game": {"id": "connect4", "params": {}},
        "agent": {"id": "dqn", "params": {}},
        "train": {"total_steps": 100, "opponent_sampler": {"type": "random", "params": {}}, "callbacks": []},
        "seed": 42,
    }
    cfg = AppConfig.from_dict(data)
    assert cfg.seed == 42


def test_opponent_sampler_required():
    data = {
        "game": {"id": "connect4", "params": {}},
        "agent": {"id": "dqn", "params": {}},
        "train": {"total_steps": 100, "callbacks": []},
    }
    with pytest.raises(ValueError, match="opponent_sampler is required"):
        AppConfig.from_dict(data)


def test_load_config():
    yaml_content = """
seed: 123
game:
  id: connect4
  params: { rows: 6, cols: 7 }
agent:
  id: dqn
  params: { learning_rate: 0.0005 }
train:
  total_steps: 500
  opponent_sampler:
    type: random
    params: {}
  callbacks:
    - type: checkpoint
      params: { prefix: model }
eval:
  enabled: true
  num_episodes: 20
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        path = Path(f.name)
    try:
        cfg = load_config(path)
        assert cfg.seed == 123
        assert cfg.game.id == "connect4"
        assert cfg.agent.params["learning_rate"] == 0.0005
        assert cfg.train.total_steps == 500
        assert len(cfg.train.callbacks) == 1
        assert cfg.train.callbacks[0].type == "checkpoint"
        assert cfg.train.callbacks[0].params["prefix"] == "model"
        assert cfg.eval.enabled is True
        assert cfg.eval.num_episodes == 20
    finally:
        path.unlink()

