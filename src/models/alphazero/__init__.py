"""AlphaZero neural network architectures."""

from .base_network import BaseAlphaZeroNetwork
from .connect4_network import Connect4AlphaZeroNetwork
from .tictactoe_network import TicTacToeAlphaZeroNetwork

__all__ = [
    "BaseAlphaZeroNetwork",
    "Connect4AlphaZeroNetwork",
    "TicTacToeAlphaZeroNetwork",
]
