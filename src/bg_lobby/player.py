"""Per-player recruitment state (shop phase); shared across rulesets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Tuple

from src.bg_catalog.patch_context import DEFAULT_PATCH_DIR, load_patch_context
from src.bg_core.minion import Minion, Race
from src.envs.minibg.actions import MAX_SHOP_SLOTS

__all__ = [
    "Minion",
    "PlayerState",
    "PlayerPhase",
    "Race",
    "ROTATION_SHOP_TRIBES",
    "CNT_ACTIVE_SHOP_TRIBES",
    "PendingChoiceKind",
    "PendingChoice",
    "CasterKind",
    "CasterRef",
]


class PlayerPhase(IntEnum):
    SHOP = 0
    DONE = 1


class PendingChoiceKind(IntEnum):
    DISCOVER_MURLOC = 0
    ADAPT = 1
    TRIPLE_REWARD_DISCOVER = 2
    TRANSFORM_SHOP_MINION = 3


@dataclass
class PendingChoice:
    """Player must pick one of ``options`` (three card_ids or three adapt keys)."""

    kind: PendingChoiceKind
    options: Tuple[str, str, str]
    extra_modals_after: int
    options_pool_reserved: bool = False
    transform_board_idx: Optional[int] = None


class CasterKind(IntEnum):
    NONE = 0
    BOARD = 1
    HAND = 2
    HERO = 3


@dataclass(frozen=True)
class CasterRef:
    """Who triggered a shop effect (replay / RL bookkeeping)."""

    kind: CasterKind
    board_idx: Optional[int] = None
    hand_idx: Optional[int] = None


# Layout/test defaults for the pinned 36393 package — not a runtime fallback path.
_DEFAULT_PATCH = load_patch_context(str(DEFAULT_PATCH_DIR))
ROTATION_SHOP_TRIBES: Tuple[Race, ...] = _DEFAULT_PATCH.meta.rotation_tribes
CNT_ACTIVE_SHOP_TRIBES = _DEFAULT_PATCH.meta.cnt_active_shop_tribes


@dataclass
class PlayerState:
    health: int
    gold: int
    tavern_tier: int
    next_tier_up_cost: int
    board: List[Minion]
    shop: List[Optional[Minion]]
    hand: List[Optional[Minion]]
    phase: PlayerPhase
    shop_actions_used: int
    shop_freeze_next_round: bool = False
    shop_frozen: Tuple[bool, ...] = (False,) * MAX_SHOP_SLOTS
    upgrade_cost_delta: int = 0
    next_roll_cost_override: Optional[int] = None
    last_combat_won: bool = False
    last_opponent_board: Tuple[Minion, ...] = ()
    shop_elemental_bonus: int = 0
    elementals_played: int = 0
    pirates_bought_this_turn: int = 0
    hero_damage_taken_total: int = 0
    pogo_hoppers_played: int = 0
    pending_choice: Optional["PendingChoice"] = None
    placed_minion_board_index: Optional[int] = None
    placed_minion_pending_after: Optional["Minion"] = None
    triple_reward_discover_pending: bool = False
    triple_reward_spell_tier: int = 0

    @property
    def shopping_finished(self) -> bool:
        return self.phase == PlayerPhase.DONE
