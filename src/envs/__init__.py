"""Environment modules."""

from .base import StepResult, TurnBasedEnv
from .connect4 import Connect4Env
from .othello import OthelloEnv
from .reward_config import RewardConfig
from .toy import ChainMDP
from ..registry import list_games, register_game

if "connect4" not in list_games():
    register_game("connect4", Connect4Env)

if "othello" not in list_games():
    register_game("othello", OthelloEnv)

__all__ = ["Connect4Env", "OthelloEnv", "RewardConfig", "StepResult", "TurnBasedEnv", "ChainMDP"]

