"""Environment modules."""

from .base import StepResult, TurnBasedEnv
from .connect4_env import Connect4Env
from .reward_config import RewardConfig
from ..registry import list_games, register_game

if "connect4" not in list_games():
    register_game("connect4", Connect4Env)

__all__ = ["Connect4Env", "RewardConfig", "StepResult", "TurnBasedEnv"]

