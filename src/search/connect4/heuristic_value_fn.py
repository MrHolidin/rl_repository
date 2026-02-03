"""Trivial heuristic value function for Connect4 minimax (terminal only)."""

from __future__ import annotations

from src.envs.connect4 import Connect4Game, Connect4State, connect4_terminal_evaluator
from src.games.turn_based_game import TurnBasedGame
from ..value_fn import StateValueFn


class Connect4HeuristicValueFn(StateValueFn[Connect4State]):
    """Uses connect4_terminal_evaluator for terminal states, 0.0 otherwise."""

    def evaluate(
        self,
        game: TurnBasedGame[Connect4State],
        state: Connect4State,
    ) -> float:
        if not game.is_terminal(state):
            return 0.0
        current_token = game.current_player(state)
        return connect4_terminal_evaluator(game, state, root_player=current_token)
