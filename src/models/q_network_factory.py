"""Factory utilities for building Q-value networks based on observation type and game."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..features.observation_builder import ObservationType
from .base_dqn_network import BaseDQNNetwork
from .dueling_utils import dueling_aggregate


# ---------------------------------------------------------------------------
# Registry for game-specific DQN networks
# ---------------------------------------------------------------------------

_NETWORK_REGISTRY_BY_CLASS: Dict[str, Type[BaseDQNNetwork]] = {}


def register_network(network_class: Type[BaseDQNNetwork]) -> Type[BaseDQNNetwork]:
    """
    Register a game-specific DQN network class.
    
    Args:
        network_class: The network class to register.
        
    Returns:
        The same class (for use as decorator).
    """
    class_name = network_class.get_class_name()
    _NETWORK_REGISTRY_BY_CLASS[class_name] = network_class
    return network_class


def get_registered_networks() -> Dict[str, Type[BaseDQNNetwork]]:
    """Return a copy of the network registry (by class name)."""
    return dict(_NETWORK_REGISTRY_BY_CLASS)


def get_network_class(class_name: str) -> Optional[Type[BaseDQNNetwork]]:
    """
    Get network class by its class name.
    
    Args:
        class_name: Name of the network class (e.g., "Connect4DQN").
        
    Returns:
        The network class if registered, None otherwise.
    """
    return _NETWORK_REGISTRY_BY_CLASS.get(class_name)


def create_network_from_checkpoint(
    class_name: str,
    kwargs: Dict,
) -> BaseDQNNetwork:
    """
    Create a network instance from checkpoint data.
    
    Args:
        class_name: Name of the network class.
        kwargs: Constructor arguments.
        
    Returns:
        Network instance.
        
    Raises:
        ValueError: If network class is not registered.
    """
    network_class = get_network_class(class_name)
    if network_class is None:
        raise ValueError(
            f"Network class '{class_name}' not found in registry. "
            f"Available: {list(_NETWORK_REGISTRY_BY_CLASS.keys())}"
        )
    return network_class(**kwargs)


# ---------------------------------------------------------------------------
# Generic network implementations for fallback
# ---------------------------------------------------------------------------

def _ensure_default_conv_layers(config: Optional[Dict]) -> List[Dict[str, int]]:
    if config is None or "conv_layers" not in config:
        return [
            {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1},
            {"out_channels": 128, "kernel_size": 3, "stride": 1, "padding": 1},
        ]
    return list(config["conv_layers"])


def _ensure_default_fc_layers(config: Optional[Dict], default: Sequence[int]) -> List[int]:
    if config is None or "fc_layers" not in config:
        return list(default)
    return list(config["fc_layers"])


class ConvQNetwork(nn.Module):
    """Convolutional Q-network suitable for board-like observations."""

    def __init__(
        self,
        observation_shape: Tuple[int, int, int],
        num_actions: int,
        conv_layers_config: List[Dict[str, int]],
        fc_layers: List[int],
        dueling: bool,
    ) -> None:
        super().__init__()
        in_channels = observation_shape[0]
        layers: List[nn.Module] = []

        current_channels = in_channels
        for layer_cfg in conv_layers_config:
            layers.append(
                nn.Conv2d(
                    current_channels,
                    layer_cfg["out_channels"],
                    kernel_size=layer_cfg.get("kernel_size", 3),
                    stride=layer_cfg.get("stride", 1),
                    padding=layer_cfg.get("padding", 1),
                )
            )
            layers.append(nn.ReLU())
            current_channels = layer_cfg["out_channels"]

        self.conv = nn.Sequential(*layers)

        rows, cols = observation_shape[1], observation_shape[2]
        self._flatten_size = current_channels * rows * cols

        if dueling:
            self.value_stream = _build_mlp(self._flatten_size, fc_layers, 1)
            self.adv_stream = _build_mlp(self._flatten_size, fc_layers, num_actions)
        else:
            self.fc = _build_mlp(self._flatten_size, fc_layers, num_actions)

        self.dueling = dueling

    def forward(self, x: torch.Tensor, legal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)

        if self.dueling:
            value = self.value_stream(x)
            advantage = self.adv_stream(x)
            return dueling_aggregate(value, advantage, legal_mask)

        return self.fc(x)


class MLPQNetwork(nn.Module):
    """Multi-layer perceptron Q-network for vector observations."""

    def __init__(self, observation_shape: Tuple[int, ...], num_actions: int, hidden_layers: Sequence[int], dueling: bool) -> None:
        super().__init__()
        input_dim = int(torch.prod(torch.tensor(observation_shape)))

        if dueling:
            self.value_stream = _build_mlp(input_dim, hidden_layers, 1)
            self.adv_stream = _build_mlp(input_dim, hidden_layers, num_actions)
        else:
            self.body = _build_mlp(input_dim, hidden_layers, num_actions)

        self.dueling = dueling

    def forward(self, x: torch.Tensor, legal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        if self.dueling:
            value = self.value_stream(x)
            advantage = self.adv_stream(x)
            return dueling_aggregate(value, advantage, legal_mask)
        return self.body(x)


def _build_mlp(input_dim: int, hidden_layers: Sequence[int], output_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_layers:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Main factory function
# ---------------------------------------------------------------------------

def build_q_network(
    observation_type: ObservationType,
    observation_shape: Tuple[int, ...],
    num_actions: int,
    dueling: bool = False,
    model_config: Optional[Dict] = None,
    network_class: Optional[str] = None,
) -> nn.Module:
    """
    Construct a Q-network based on the observation type and optional network class.
    
    Args:
        observation_type: Type of observation ("board" or "vector").
        observation_shape: Shape of the observation tensor.
        num_actions: Number of discrete actions.
        dueling: Whether to use dueling architecture.
        model_config: Optional dictionary with model hyper-parameters.
        network_class: Optional network class name to use (e.g., "Connect4DQN").
        
    Returns:
        A Q-network module.
    """
    # Try to use specified network class if available
    if network_class is not None:
        cls = get_network_class(network_class)
        if cls is not None:
            return _build_game_specific_network(
                cls,
                observation_shape,
                num_actions,
                dueling,
                model_config,
            )
    
    # Auto-detect game-specific networks by board shape
    if observation_type == "board" and len(observation_shape) == 3:
        in_channels, rows, cols = observation_shape
        
        # Connect4: 6x7 board
        if rows == 6 and cols == 7:
            cls = get_network_class("Connect4DQN")
            if cls is not None:
                return _build_game_specific_network(
                    cls,
                    observation_shape,
                    num_actions,
                    dueling,
                    model_config,
                )
        
        # Othello: 8x8 board
        if rows == 8 and cols == 8:
            cls = get_network_class("OthelloDQN")
            if cls is not None:
                return _build_game_specific_network(
                    cls,
                    observation_shape,
                    num_actions,
                    dueling,
                    model_config,
                )
    
    # Fallback to generic networks
    if observation_type == "board":
        conv_layers = _ensure_default_conv_layers(model_config)
        fc_layers = _ensure_default_fc_layers(model_config, default=[256])
        return ConvQNetwork(observation_shape, num_actions, conv_layers, fc_layers, dueling)

    if observation_type == "vector":
        hidden_layers = _ensure_default_fc_layers(model_config, default=[256, 256])
        return MLPQNetwork(observation_shape, num_actions, hidden_layers, dueling)

    raise ValueError(f"Unsupported observation type '{observation_type}'.")


def _build_game_specific_network(
    network_class: Type[BaseDQNNetwork],
    observation_shape: Tuple[int, ...],
    num_actions: int,
    dueling: bool,
    model_config: Optional[Dict],
) -> BaseDQNNetwork:
    """
    Build a game-specific network with appropriate parameters.
    
    Args:
        network_class: The specialized network class.
        observation_shape: Shape of the observation tensor.
        num_actions: Number of discrete actions.
        dueling: Whether to use dueling architecture.
        model_config: Optional dictionary with model hyper-parameters.
        
    Returns:
        Instantiated game-specific network.
    """
    class_name = network_class.get_class_name()
    
    # For board-based games, extract dimensions
    if len(observation_shape) == 3:
        in_channels, rows, cols = observation_shape
    else:
        in_channels, rows, cols = None, None, None
    
    # Build kwargs based on network type
    if class_name == "OthelloDQN":
        # OthelloDQN: uses board_size instead of rows/cols
        kwargs = {
            "board_size": rows if rows is not None else 8,
            "in_channels": in_channels if in_channels is not None else 2,
            "num_actions": num_actions,
            "dueling": dueling,
        }
    else:
        # Connect4DQN and others: use rows/cols
        kwargs = {
            "num_actions": num_actions,
            "dueling": dueling,
        }
        if in_channels is not None:
            kwargs["in_channels"] = in_channels
        if rows is not None:
            kwargs["rows"] = rows
        if cols is not None:
            kwargs["cols"] = cols
    
    # Apply any model_config overrides
    if model_config is not None:
        for key, value in model_config.items():
            if key in ["rows", "cols", "in_channels", "board_size", 
                       "trunk_channels", "num_res_blocks", "use_coord_channels",
                       "head_hidden", "adv_hidden", "val_hidden"]:
                kwargs[key] = value
    
    return network_class(**kwargs)


# ---------------------------------------------------------------------------
# Auto-register game-specific networks on import
# ---------------------------------------------------------------------------

def _auto_register_networks():
    """Register all known game-specific networks."""
    from .connect4_dqn import Connect4DQN, Connect4QRDQN
    from .othello_dqn import OthelloDQN, OthelloQRDQN
    register_network(Connect4DQN)
    register_network(Connect4QRDQN)
    register_network(OthelloDQN)
    register_network(OthelloQRDQN)


_auto_register_networks()
