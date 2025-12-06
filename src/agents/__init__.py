"""Agent modules."""

from .base_agent import BaseAgent
from .random_agent import RandomAgent
from .heuristic_agent import HeuristicAgent
from .smart_heuristic_agent import SmartHeuristicAgent
from .qlearning_agent import QLearningAgent
from .dqn_agent import DQNAgent
from .ppo_agent import PPOAgent
from ..features.action_space import DiscreteActionSpace
from ..features.observation_builder import BoardChannels
from ..models import Connect4DQN
from ..registry import list_agents, register_agent

if "random" not in list_agents():
    register_agent("random", RandomAgent)
if "heuristic" not in list_agents():
    register_agent("heuristic", HeuristicAgent)
if "smart_heuristic" not in list_agents():
    register_agent("smart_heuristic", SmartHeuristicAgent)
if "qlearning" not in list_agents():
    register_agent("qlearning", QLearningAgent)
if "dqn" not in list_agents():
    def _dqn_factory(**kwargs):
        # If network is already provided, use it directly
        if "network" in kwargs:
            # Clean up old-style params
            kwargs.pop("observation_shape", None)
            kwargs.pop("observation_type", None)
            kwargs.pop("network_type", None)
            return DQNAgent(**kwargs)
        
        # Extract network-related params
        obs_shape = kwargs.pop("observation_shape", None)
        kwargs.pop("observation_type", None)  # not used anymore
        network_type = kwargs.pop("network_type", "dqn")
        
        action_space = kwargs.get("action_space")
        num_actions = kwargs.get("num_actions")

        if action_space is None and num_actions is not None:
            action_space = DiscreteActionSpace(num_actions)
            kwargs["action_space"] = action_space

        # Default to Connect4 dimensions
        if obs_shape is None:
            builder = BoardChannels(board_shape=(6, 7))
            obs_shape = builder.observation_shape
        
        if num_actions is None:
            num_actions = 7
            kwargs["num_actions"] = num_actions
        
        if action_space is None:
            action_space = DiscreteActionSpace(num_actions)
            kwargs["action_space"] = action_space

        # Create network
        in_channels = obs_shape[0] if len(obs_shape) == 3 else 3
        rows = obs_shape[1] if len(obs_shape) == 3 else 6
        cols = obs_shape[2] if len(obs_shape) == 3 else 7
        dueling = network_type == "dueling_dqn"
        
        network = Connect4DQN(
            rows=rows,
            cols=cols,
            in_channels=in_channels,
            num_actions=num_actions,
            dueling=dueling,
        )
        
        return DQNAgent(network=network, **kwargs)
    register_agent("dqn", _dqn_factory)
if "ppo" not in list_agents():
    def _ppo_factory(**kwargs):
        obs_shape = kwargs.get("observation_shape")
        obs_type = kwargs.get("observation_type")
        action_space = kwargs.get("action_space")
        num_actions = kwargs.get("num_actions")

        if action_space is None and num_actions is not None:
            action_space = DiscreteActionSpace(num_actions)
            kwargs["action_space"] = action_space

        if obs_shape is None or obs_type is None or num_actions is None:
            builder = BoardChannels(board_shape=(6, 7))
            default_action_space = action_space or DiscreteActionSpace(n=7)
            kwargs.setdefault("observation_shape", builder.observation_shape)
            kwargs.setdefault("observation_type", builder.observation_type)
            kwargs.setdefault("action_space", default_action_space)
            kwargs.setdefault("num_actions", default_action_space.size)
        else:
            kwargs.setdefault("observation_shape", obs_shape)
            kwargs.setdefault("observation_type", obs_type)
            if action_space is None:
                kwargs.setdefault("action_space", DiscreteActionSpace(num_actions))
            kwargs.setdefault("num_actions", num_actions)

        return PPOAgent(**kwargs)
    register_agent("ppo", _ppo_factory)

__all__ = [
    "BaseAgent",
    "RandomAgent",
    "HeuristicAgent",
    "SmartHeuristicAgent",
    "QLearningAgent",
    "DQNAgent",
    "PPOAgent",
]
