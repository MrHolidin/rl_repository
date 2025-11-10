"""Tests for agents."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs import Connect4Env
from src.agents import RandomAgent, HeuristicAgent, QLearningAgent, DQNAgent
from src.features.action_space import DiscreteActionSpace
from src.features.observation_builder import BoardChannels


def test_random_agent():
    """Test random agent."""
    env = Connect4Env(rows=6, cols=7)
    agent = RandomAgent(seed=42)
    
    obs = env.reset()
    legal_actions = env.get_legal_actions()
    
    action = agent.select_action(obs, legal_actions)
    assert action in legal_actions


def test_heuristic_agent():
    """Test heuristic agent."""
    env = Connect4Env(rows=6, cols=7)
    agent = HeuristicAgent(seed=42)
    
    obs = env.reset()
    legal_actions = env.get_legal_actions()
    
    action = agent.select_action(obs, legal_actions)
    assert action in legal_actions


def test_qlearning_agent():
    """Test Q-learning agent."""
    env = Connect4Env(rows=6, cols=7)
    agent = QLearningAgent(seed=42)
    
    obs = env.reset()
    legal_actions = env.get_legal_actions()
    
    # Select action
    action = agent.select_action(obs, legal_actions)
    assert action in legal_actions
    
    # Step environment
    next_obs, reward, done, info = env.step(action)
    
    # Observe transition
    agent.observe((obs, action, reward, next_obs, done, info))
    
    # Check that Q-table was updated
    assert len(agent.q_table) > 0


def test_dqn_agent():
    """Test DQN agent."""
    env = Connect4Env(rows=6, cols=7)
    builder = BoardChannels(board_shape=(6, 7))
    action_space = DiscreteActionSpace(n=7)
    agent = DQNAgent(
        observation_shape=builder.observation_shape,
        observation_type=builder.observation_type,
        num_actions=action_space.size,
        seed=42,
        action_space=action_space,
    )
    
    obs = env.reset()
    legal_actions = env.get_legal_actions()
    legal_mask = env.legal_actions_mask.astype(bool)
    
    # Select action
    action = agent.select_action(obs, legal_actions)
    assert action in legal_actions
    
    # Step environment
    next_obs, reward, done, info = env.step(action)
    if done:
        next_legal_mask = np.zeros_like(legal_mask)
    else:
        next_legal_mask = env.legal_actions_mask.astype(bool)
    
    # Observe transition
    agent.observe((obs, action, reward, next_obs, done, info, legal_mask, next_legal_mask))
    
    # Check that replay buffer has samples
    assert len(agent.replay_buffer) > 0


def test_agent_train_eval_mode():
    """Test agent train/eval mode switching."""
    builder = BoardChannels(board_shape=(6, 7))
    action_space = DiscreteActionSpace(n=7)
    agent = DQNAgent(
        observation_shape=builder.observation_shape,
        observation_type=builder.observation_type,
        num_actions=action_space.size,
        seed=42,
        action_space=action_space,
    )
    
    # Initially in training mode
    assert agent.training
    
    # Switch to eval mode
    agent.eval()
    assert not agent.training
    
    # Switch back to train mode
    agent.train()
    assert agent.training


if __name__ == "__main__":
    pytest.main([__file__])

