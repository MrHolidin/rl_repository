"""Connect4 evaluation functions for search algorithms."""

from __future__ import annotations

from typing import Optional

from .state import Connect4State
from src.games.turn_based_game import TurnBasedGame


def connect4_terminal_evaluator(
    game: TurnBasedGame[Connect4State],
    state: Connect4State,
    root_player: int,
) -> float:
    if not game.is_terminal(state):
        return 0.0

    winner = game.winner(state)
    if winner is None or winner == 0:
        return 0.0
    if winner == root_player:
        return 1.0
    if winner == -root_player:
        return -1.0
    return 0.0
