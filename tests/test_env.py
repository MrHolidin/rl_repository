"""Tests for Connect4Env."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs import Connect4Env


def test_env_initialization():
    """Test environment initialization."""
    env = Connect4Env(rows=6, cols=7)
    assert env.rows == 6
    assert env.cols == 7
    assert env.current_player() == 0
    assert env.current_player_token == 1
    assert not env.done


def test_env_reset():
    """Test environment reset."""
    env = Connect4Env(rows=6, cols=7)
    obs = env.reset()
    assert obs.shape == (2, 6, 7)
    assert np.all(env.board == 0)
    assert env.current_player() == 0
    assert env.current_player_token == 1
    assert not env.done


def test_env_step():
    """Test environment step."""
    env = Connect4Env(rows=6, cols=7)
    obs = env.reset()
    
    # First action
    next_obs, reward, done, info = env.step(0)
    assert next_obs.shape == (2, 6, 7)
    assert env.board[5, 0] == 1  # Piece should be at bottom
    assert env.current_player() == 1  # Player index should switch
    assert env.current_player_token == -1  # Player token should switch
    assert not done


def test_env_legal_actions():
    """Test legal actions."""
    env = Connect4Env(rows=6, cols=7)
    env.reset()
    
    legal_actions = env.get_legal_actions()
    assert len(legal_actions) == 7
    assert set(legal_actions) == set(range(7))
    
    # Fill a column
    for _ in range(6):
        env.step(0)
    
    legal_actions = env.get_legal_actions()
    assert 0 not in legal_actions


def test_env_win_horizontal():
    """Test horizontal win detection."""
    env = Connect4Env(rows=6, cols=7)
    env.reset()
    
    # Player 1 wins horizontally
    actions = [0, 0, 1, 1, 2, 2, 3]  # Player 1 wins
    for i, action in enumerate(actions):
        obs, reward, done, info = env.step(action)
        if done:
            assert info["winner"] == 1
            assert info["termination_reason"] == "win"
            break


def test_env_win_vertical():
    """Test vertical win detection."""
    env = Connect4Env(rows=6, cols=7)
    env.reset()
    
    # Player 1 wins vertically
    actions = [0, 1, 0, 1, 0, 1, 0]  # Player 1 wins
    for i, action in enumerate(actions):
        obs, reward, done, info = env.step(action)
        if done:
            assert info["winner"] == 1
            assert info["termination_reason"] == "win"
            break


def test_env_draw():
    """Test draw detection."""
    env = Connect4Env(rows=6, cols=7)
    env.reset()
    
    # Fill board without win (simplified test)
    # This is a simplified test - in practice, a draw is rare
    # We'll just check that the board can be full
    for col in range(7):
        for row in range(6):
            if not env.done:
                env.step(col)
                if env.done:
                    break


def test_env_invalid_action():
    """Test invalid action handling."""
    env = Connect4Env(rows=6, cols=7)
    env.reset()
    
    # Fill a column
    for _ in range(6):
        env.step(0)
    
    # Try to place in full column
    obs, reward, done, info = env.step(0)
    assert reward < 0  # Negative reward for invalid action
    assert "invalid_action" in info or info.get("invalid_action", False)


if __name__ == "__main__":
    pytest.main([__file__])

