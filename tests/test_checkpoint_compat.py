"""Checkpoint compatibility: load canonical checkpoint, verify inference matches probe."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.agents.dqn_agent import DQNAgent
from src.training.canonical_checkpoint import train_and_probe

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CKPT_PATH = FIXTURES / "canonical_dqn.pt"
PROBE_PATH = FIXTURES / "canonical_dqn_probe.json"


def _load_probe() -> list:
    with open(PROBE_PATH) as f:
        return json.load(f)["probes"]


def test_checkpoint_produces_same_inference():
    """Load canonical checkpoint and verify actions match probe data."""
    if os.environ.get("UPDATE_CANONICAL") == "1":
        pytest.skip("Skipped when UPDATE_CANONICAL=1")
    if not CKPT_PATH.exists():
        pytest.skip(f"Canonical checkpoint not found. Run: python scripts/generate_canonical_checkpoint.py")

    agent = DQNAgent.load(str(CKPT_PATH), load_optimizer=False)
    agent.eval()

    probes = _load_probe()
    for i, p in enumerate(probes):
        obs = np.array(p["obs"], dtype=np.float32)
        legal = np.array(p["legal_mask"], dtype=bool)
        expected = p["expected_action"]
        action = agent.act(obs, legal_mask=legal, deterministic=True)
        assert action == expected, f"Probe {i}: expected {expected}, got {action}"


def test_training_deterministic():
    """Train twice with same seed; probe actions must match."""
    _, actions1 = train_and_probe(seed=42, steps=300)
    _, actions2 = train_and_probe(seed=42, steps=300)
    assert actions1 == actions2, f"Probe actions differ: {actions1} vs {actions2}"


def test_training_matches_canonical():
    """Train from scratch; probe actions must match canonical file (unchanged during test)."""
    if os.environ.get("UPDATE_CANONICAL") == "1":
        pytest.skip("Skipped when UPDATE_CANONICAL=1")
    if not PROBE_PATH.exists():
        pytest.skip(f"Canonical probe not found. Run: UPDATE_CANONICAL=1 pytest {__file__} -k test_update_canonical")

    _, actions = train_and_probe(seed=42, steps=300)
    probes = _load_probe()
    expected = [p["expected_action"] for p in probes]
    assert actions == expected, f"Training result differs from canonical: {actions} vs {expected}"


def test_update_canonical():
    """Regenerate canonical checkpoint and probe. Run with UPDATE_CANONICAL=1 pytest ..."""
    if os.environ.get("UPDATE_CANONICAL") != "1":
        pytest.skip("Set UPDATE_CANONICAL=1 to regenerate canonical files")

    subprocess.run(
        [sys.executable, "scripts/generate_canonical_checkpoint.py"],
        cwd=Path(__file__).resolve().parent.parent,
        check=True,
    )
    assert CKPT_PATH.exists()
    assert PROBE_PATH.exists()
