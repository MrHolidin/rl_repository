"""Trivial heuristic value function for Connect4 minimax (terminal only)."""

from __future__ import annotations

from src.envs.connect4 import Connect4Game, Connect4State
from src.games.turn_based_game import TurnBasedGame
from src.games.state_evaluator import terminal_value
from ..value_fn import StateValueFn


class Connect4HeuristicValueFn(StateValueFn[Connect4State]):
    """Returns terminal_value for terminal states, 0.0 otherwise."""

    def evaluate(
        self,
        game: TurnBasedGame[Connect4State],
        state: Connect4State,
    ) -> float:
        if not game.is_terminal(state):
            return 0.0
        return terminal_value(game, state, root_player=game.current_player(state))
