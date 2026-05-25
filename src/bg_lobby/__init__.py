"""Match / lobby orchestration (pairing, round advance, combat scheduling)."""

from .player import (
    CasterKind,
    CasterRef,
    Minion,
    PendingChoice,
    PendingChoiceKind,
    PlayerPhase,
    PlayerState,
    Race,
)
from .match_types import GHOST_OPPONENT_ID, CombatMatch, EliminatedSnapshot
from .pairing import (
    COOLDOWN_COMBAT_ROUNDS,
    DEFAULT_LOBBY_SIZE,
    build_round_robin_schedule,
    compute_pairings,
    record_combat_opponent,
)
from .shared_pool import (
    SharedCardPool,
    build_initial_shared_pool,
    copies_for_minion,
)
from .shop_order import sample_shop_turn_order

__all__ = [
    "SharedCardPool",
    "build_initial_shared_pool",
    "copies_for_minion",
    "COOLDOWN_COMBAT_ROUNDS",
    "DEFAULT_LOBBY_SIZE",
    "GHOST_OPPONENT_ID",
    "CombatMatch",
    "EliminatedSnapshot",
    "build_round_robin_schedule",
    "compute_pairings",
    "record_combat_opponent",
    "CasterKind",
    "CasterRef",
    "Minion",
    "PendingChoice",
    "PendingChoiceKind",
    "PlayerPhase",
    "PlayerState",
    "Race",
    "sample_shop_turn_order",
]
