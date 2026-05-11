from pathlib import Path

import pytest

from src.training.callbacks.metrics_file import MetricsFileCallback
from src.training.metrics_presets import (
    LEGACY_DQN_METRICS_FIELDS,
    METRICS_PRESET_PPO,
    resolve_metrics_csv_fieldnames,
)


def test_resolve_auto_ppo():
    cols = resolve_metrics_csv_fieldnames("ppo", preset="auto")
    assert "policy_loss" in cols
    assert "avg_q" not in cols
    assert cols[:4] == ("step", "episode", "epsilon", "learning_rate")


def test_resolve_auto_dqn():
    cols = resolve_metrics_csv_fieldnames("dqn", preset="auto")
    assert "avg_q" in cols
    assert "clip_frac" not in cols


def test_resolve_preset_explicit():
    assert "approx_kl" in resolve_metrics_csv_fieldnames("dqn", preset="ppo")


def test_resolve_explicit_columns_override():
    cols = resolve_metrics_csv_fieldnames("anything", preset="auto", columns=["a", "b"])
    assert cols == ("a", "b")


def test_resolve_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown metrics_file preset"):
        resolve_metrics_csv_fieldnames("x", preset="bogus")


def test_metrics_file_callback_custom_header(tmp_path):
    path = Path(tmp_path)
    fields = ("step", "episode", "epsilon", "loss")
    cb = MetricsFileCallback(path, interval=1, fieldnames=fields)
    cb.on_train_begin(None)  # type: ignore[arg-type]
    assert cb.path.read_text().strip().split(",") == list(fields)
