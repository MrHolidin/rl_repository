"""Self-play module for training agents against their previous versions."""

from .opponent_pool import OpponentPool, FrozenAgentInfo, SelfPlayConfig, SelfPlayOpponent

__all__ = ["OpponentPool", "FrozenAgentInfo", "SelfPlayConfig", "SelfPlayOpponent"]

