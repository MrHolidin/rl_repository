"""Self-play training components."""

from .league_policy import OpponentKind, decide_opponent_kind, pfsp_sample
from .league_state import LeagueController, LeagueSnapshot, SLOT_CURRENT, SLOT_SCRIPTED
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
    "OpponentKind",
    "decide_opponent_kind",
    "pfsp_sample",
    "LeagueController",
    "LeagueSnapshot",
    "SLOT_CURRENT",
    "SLOT_SCRIPTED",
]
