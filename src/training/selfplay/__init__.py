"""Self-play training components."""

from .opponent_pool import OpponentPool, FrozenAgentInfo, SelfPlayConfig, SelfPlayOpponent

__all__ = ["OpponentPool", "FrozenAgentInfo", "SelfPlayConfig", "SelfPlayOpponent"]
