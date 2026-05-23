"""Self-play training components."""

from .game_record import GameRecord, SLOT_CURRENT, SLOT_SCRIPTED
from .league_state import LeagueController, LeagueSnapshot
from .rating_system import EmaPairwiseRating, RatingSystem, TrueSkillRating, make_rating_system
from .slot_registry import SlotRegistry
from .opponent_pool import (
    FrozenAgentInfo,
    OpponentPool,
    ScriptedOpponentsSpec,
    SelfPlayConfig,
    SelfPlayOpponent,
)

__all__ = [
    "GameRecord",
    "OpponentPool",
    "FrozenAgentInfo",
    "ScriptedOpponentsSpec",
    "SelfPlayConfig",
    "SelfPlayOpponent",
    "LeagueController",
    "LeagueSnapshot",
    "EmaPairwiseRating",
    "RatingSystem",
    "TrueSkillRating",
    "SlotRegistry",
    "make_rating_system",
    "SLOT_CURRENT",
    "SLOT_SCRIPTED",
]
