"""Environment modules."""

from .base import StepResult, TurnBasedEnv
from .connect4 import Connect4Env
from .minibg import MiniBGEnv
from .othello import OthelloEnv
from .reward_config import RewardConfig
from .toy import ChainMDP
from ..registry import list_games, register_game

if "connect4" not in list_games():
    register_game("connect4", Connect4Env)

if "othello" not in list_games():
    register_game("othello", OthelloEnv)

if "minibg" not in list_games():
    register_game("minibg", MiniBGEnv)

__all__ = [
    "Connect4Env",
    "MiniBGEnv",
    "OthelloEnv",
    "RewardConfig",
    "StepResult",
    "TurnBasedEnv",
    "ChainMDP",
]

