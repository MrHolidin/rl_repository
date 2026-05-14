"""Self-play training components."""

from .opponent_pool import (
    FrozenAgentInfo,
    OpponentPool,
    ScriptedOpponentsSpec,
    SelfPlayConfig,
    SelfPlayOpponent,
)

__all__ = [
    "OpponentPool",
    "FrozenAgentInfo",
    "ScriptedOpponentsSpec",
    "SelfPlayConfig",
    "SelfPlayOpponent",
]
