"""Observation builders for transforming raw environment state into model inputs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

ObservationType = Literal["board", "vector"]


class ObservationBuilder(ABC):
    """Abstract base class for converting raw environment state into observations."""

    def __init__(self, observation_type: ObservationType) -> None:
        self._observation_type = observation_type

    @property
    def observation_type(self) -> ObservationType:
        """Type of observation produced by the builder."""
        return self._observation_type

    @property
    @abstractmethod
    def observation_shape(self) -> Tuple[int, ...]:
        """Shape of the produced observation."""

    @abstractmethod
    def build(self, raw_state: Any) -> np.ndarray:
        """Produce an observation tensor from the raw environment state."""


class BoardChannels(ObservationBuilder):
    """Build observations for board-like, two-player games using multi-channel encoding."""

    def __init__(
        self,
        board_shape: Tuple[int, int],
        include_last_move: bool = False,
        include_legal_moves: bool = False,
        dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__("board")
        self._rows, self._cols = board_shape
        self._include_last_move = include_last_move
        self._include_legal_moves = include_legal_moves
        self._dtype = dtype

        channels = 3  # current player pieces, opponent pieces, current player indicator
        if include_last_move:
            channels += 1
        if include_legal_moves:
            channels += 1
        self._observation_shape = (channels, self._rows, self._cols)

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return self._observation_shape

    def build(self, raw_state: Dict[str, Any]) -> np.ndarray:
        board = raw_state["board"]
        current_player_token: int = raw_state["current_player_token"]
        last_move: Optional[Tuple[int, int]] = raw_state.get("last_move")
        legal_moves_mask: Optional[np.ndarray] = raw_state.get("legal_actions_mask")

        obs = np.zeros(self._observation_shape, dtype=self._dtype)
        current_channel = (board == current_player_token).astype(self._dtype)
        opponent_channel = (board == -current_player_token).astype(self._dtype)
        turn_channel = np.full((self._rows, self._cols), 1.0 if current_player_token == 1 else 0.0, dtype=self._dtype)

        obs[0] = current_channel
        obs[1] = opponent_channel
        obs[2] = turn_channel

        channel_idx = 3

        if self._include_last_move:
            last_move_channel = np.zeros((self._rows, self._cols), dtype=self._dtype)
            if last_move is not None:
                last_move_channel[last_move] = 1.0
            obs[channel_idx] = last_move_channel
            channel_idx += 1

        if self._include_legal_moves:
            legal_channel = np.zeros((self._rows, self._cols), dtype=self._dtype)
            if legal_moves_mask is not None:
                legal_channel[0, :] = legal_moves_mask.astype(self._dtype)
            obs[channel_idx] = legal_channel

        return obs

