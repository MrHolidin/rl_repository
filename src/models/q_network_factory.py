"""Factory utilities for building Q-value networks based on observation type."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..features.observation_builder import ObservationType
from .dueling_utils import dueling_aggregate


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


def build_q_network(
    observation_type: ObservationType,
    observation_shape: Tuple[int, ...],
    num_actions: int,
    dueling: bool = False,
    model_config: Optional[Dict] = None,
) -> nn.Module:
    """Construct a Q-network based on the observation type."""
    if observation_type == "board":
        conv_layers = _ensure_default_conv_layers(model_config)
        fc_layers = _ensure_default_fc_layers(model_config, default=[256])
        return ConvQNetwork(observation_shape, num_actions, conv_layers, fc_layers, dueling)

    if observation_type == "vector":
        hidden_layers = _ensure_default_fc_layers(model_config, default=[256, 256])
        return MLPQNetwork(observation_shape, num_actions, hidden_layers, dueling)

    raise ValueError(f"Unsupported observation type '{observation_type}'.")

