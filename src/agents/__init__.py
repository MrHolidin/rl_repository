"""Agent modules."""

from .base_agent import BaseAgent
from .random_agent import RandomAgent
from .connect4 import HeuristicAgent, SmartHeuristicAgent
from .othello import OthelloHeuristicAgent
from .qlearning_agent import QLearningAgent
from .dqn.agent import DQNAgent
from .ppo_agent import PPOAgent
from ..features.action_space import DiscreteActionSpace
from ..features.observation_builder import BoardChannels
from ..models import Connect4DQN, Connect4QRDQN, OthelloDQN, OthelloQRDQN
from ..registry import list_agents, register_agent

if "random" not in list_agents():
    register_agent("random", RandomAgent)
if "heuristic" not in list_agents():
    register_agent("heuristic", HeuristicAgent)
if "smart_heuristic" not in list_agents():
    register_agent("smart_heuristic", SmartHeuristicAgent)
if "othello_heuristic" not in list_agents():
    register_agent("othello_heuristic", OthelloHeuristicAgent)
if "qlearning" not in list_agents():
    register_agent("qlearning", QLearningAgent)
if "dqn" not in list_agents():
    def _dqn_factory(**kwargs):
        # If network is already provided, use it directly
        if "network" in kwargs:
            kwargs.pop("observation_shape", None)
            kwargs.pop("observation_type", None)
            kwargs.pop("network_type", None)
            kwargs.pop("dueling", None)
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

        # Create network based on board size
        in_channels = obs_shape[0] if len(obs_shape) == 3 else 3
        rows = obs_shape[1] if len(obs_shape) == 3 else 6
        cols = obs_shape[2] if len(obs_shape) == 3 else 7
        dueling = kwargs.pop("dueling", None)
        if dueling is None:
            dueling = network_type == "dueling_dqn"
        use_distributional = kwargs.pop("use_distributional", False)
        n_quantiles = kwargs.pop("n_quantiles", 32)
        # Noisy nets: agent's use_noisy_nets controls both network's use_noisy and agent behavior
        use_noisy_nets = kwargs.get("use_noisy_nets", False)
        noisy_sigma = kwargs.pop("noisy_sigma", 0.5)
        
        if use_distributional:
            if rows == 8 and cols == 8 and num_actions == 64:
                network = OthelloQRDQN(
                    board_size=8,
                    in_channels=in_channels,
                    num_actions=num_actions,
                    n_quantiles=n_quantiles,
                )
            else:
                network = Connect4QRDQN(
                    rows=rows,
                    cols=cols,
                    in_channels=in_channels,
                    num_actions=num_actions,
                    n_quantiles=n_quantiles,
                    use_noisy=use_noisy_nets,
                    noisy_sigma=noisy_sigma,
                )
            kwargs["use_distributional"] = True
            kwargs["n_quantiles"] = n_quantiles
        else:
            if rows == 8 and cols == 8 and num_actions == 64:
                network = OthelloDQN(
                    board_size=8,
                    in_channels=in_channels,
                    num_actions=num_actions,
                )
            else:
                network = Connect4DQN(
                    rows=rows,
                    cols=cols,
                    in_channels=in_channels,
                    num_actions=num_actions,
                    dueling=dueling,
                    use_noisy=use_noisy_nets,
                    noisy_sigma=noisy_sigma,
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
