"""Tests for utilities."""

import pytest
import numpy as np
import sys
from pathlib import Path
import os
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import ReplayBuffer, MetricsLogger


def test_replay_buffer():
    """Test replay buffer."""
    buffer = ReplayBuffer(capacity=100)
    
    # Add transitions
    for i in range(10):
        obs = np.random.rand(3, 6, 7)
        action = i % 7
        reward = 0.0
        next_obs = np.random.rand(3, 6, 7)
        done = False
        buffer.push(obs, action, reward, next_obs, done)
    
    assert len(buffer) == 10
    
    # Sample batch
    batch = buffer.sample(5)
    assert len(batch) == 5
    obs_batch, actions, rewards, next_obs_batch, dones = batch
    assert obs_batch.shape[0] == 5
    assert actions.shape[0] == 5


def test_replay_buffer_capacity():
    """Test replay buffer capacity limit."""
    buffer = ReplayBuffer(capacity=10)
    
    # Add more than capacity
    for i in range(20):
        obs = np.random.rand(3, 6, 7)
        action = i % 7
        reward = 0.0
        next_obs = np.random.rand(3, 6, 7)
        done = False
        buffer.push(obs, action, reward, next_obs, done)
    
    # Should be limited to capacity
    assert len(buffer) == 10


def test_metrics_logger():
    """Test metrics logger."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = MetricsLogger(log_dir=tmpdir)
        
        # Log metrics
        logger.log("test_metric", 1.0, step=0)
        logger.log("test_metric", 2.0, step=1)
        logger.log_dict({"metric1": 1.0, "metric2": 2.0}, step=2)
        
        # Check that metrics were logged
        metrics = logger.get_metric("test_metric")
        assert len(metrics) == 2
        
        logger.close()


if __name__ == "__main__":
    pytest.main([__file__])

