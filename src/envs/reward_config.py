"""Reward configuration for Connect4 environment."""

from dataclasses import dataclass


@dataclass
class RewardConfig:
    """
    Configuration for rewards in Connect4 environment.
    
    Attributes:
        win: Reward for winning (default: 1.0)
        loss: Reward for losing (default: -1.0)
        draw: Reward for draw (default: 0.0)
        three_in_row: Reward for getting 3 in a row (default: 0.0)
        opponent_three_in_row: Penalty for opponent having 3 in a row (default: 0.0)
        invalid_action: Reward for invalid action (default: -0.1)
    """
    win: float = 1.0
    loss: float = -1.0
    draw: float = 0.0
    three_in_row: float = 0.0
    opponent_three_in_row: float = 0.0
    invalid_action: float = -0.1

