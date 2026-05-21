"""Match / lobby orchestration (pairing, round advance, combat scheduling)."""

from .player import (
    CNT_ACTIVE_SHOP_TRIBES,
    CasterKind,
    CasterRef,
    Minion,
    PendingChoice,
    PendingChoiceKind,
    PlayerPhase,
    PlayerState,
    Race,
    ROTATION_SHOP_TRIBES,
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
    POOL_SIZE_BY_TIER,
    SharedCardPool,
    build_initial_shared_pool,
    copies_for_minion,
)
from .shop_order import sample_shop_turn_order

__all__ = [
    "POOL_SIZE_BY_TIER",
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
    "CNT_ACTIVE_SHOP_TRIBES",
    "CasterKind",
    "CasterRef",
    "Minion",
    "PendingChoice",
    "PendingChoiceKind",
    "PlayerPhase",
    "PlayerState",
    "Race",
    "ROTATION_SHOP_TRIBES",
    "sample_shop_turn_order",
]
