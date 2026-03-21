"""TicTacToe observation builder helpers."""

import numpy as np

from .state import TicTacToeState
from .game import TicTacToeGame


def build_state_dict(state: TicTacToeState, game: TicTacToeGame) -> dict:
    legal = list(game.legal_actions(state))
    legal_mask = np.zeros(9, dtype=bool)
    for a in legal:
        legal_mask[a] = True
    return {
        "board": state.board,
        "current_player_token": game.current_player(state),
        "last_move": None,
        "legal_actions_mask": legal_mask,
    }
