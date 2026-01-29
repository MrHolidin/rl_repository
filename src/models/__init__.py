"""Model modules."""

from .base_dqn_network import BaseDQNNetwork
from .connect4_dqn import Connect4DQN, DQN
from .simple_mlp import SimpleMLP
from .author_critic_network import ActorCriticCNN
from .q_network_factory import (
    build_q_network,
    register_network,
    get_network_class,
    get_registered_networks,
    create_network_from_checkpoint,
)

__all__ = [
    "BaseDQNNetwork",
    "Connect4DQN",
    "DQN",  # backward compatibility alias
    "SimpleMLP",
    "ActorCriticCNN",
    "build_q_network",
    "register_network",
    "get_network_class",
    "get_registered_networks",
    "create_network_from_checkpoint",
]
