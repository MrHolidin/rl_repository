"""Per-player recruitment state (shop phase); shared across rulesets."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

from src.bg_core.hero import Hero
from src.bg_core.minion import Minion, Race
from src.envs.minibg.actions import MAX_SHOP_SLOTS

# History length for obs-side last-N-battles features. Length = 3 matches
# what real-BG shows on the opponent panel; bumping requires re-training.
BATTLE_HISTORY_LEN = 3

__all__ = [
    "BattleSnapshot",
    "Minion",
    "PlayerState",
    "PlayerPhase",
    "Race",
    "PendingChoiceKind",
    "PendingChoice",
    "CasterKind",
    "CasterRef",
]


@dataclass(frozen=True)
class BattleSnapshot:
    """Per-seat snapshot of boards as they entered combat (own/opp oriented).

    ``step_index=0`` is the pre-combat snapshot fed to the battle-prediction head.
    Future mid-battle snapshots can be appended with higher indices when
    ``simulate_battle`` is instrumented to emit them.
    """

    own_board: Tuple[Minion, ...]
    opp_board: Tuple[Minion, ...]
    step_index: int = 0


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
    free_roll_charges: int = 0
    last_combat_won: bool = False
    last_opponent_board: Tuple[Minion, ...] = ()
    shop_elemental_bonus: int = 0
    elementals_played: int = 0
    pirates_bought_this_turn: int = 0
    hero_damage_taken_total: int = 0
    pogo_hoppers_played: int = 0
    # Hero (passive power). ``None`` ⇒ classic no-hero seat (default; identical
    # to pre-hero behavior). Set at game start only when ``with_heroes=True``.
    hero: Optional[Hero] = None
    # Hero-passive counters/state (unused while ``hero is None``). These are
    # carried explicitly by ``BGLikeGame._copy_player`` (unlike the transient
    # ``upgrade_cost_delta`` / ``next_roll_cost_override``, which that copy
    # intentionally resets) so hero levers survive across shop actions.
    hero_buy_count: int = 0  # Kael'thas: every 3rd buy
    hero_rotating_tribe: Optional[Race] = None  # The Rat King: current tribe
    hero_elementals_progress: int = 0  # Chenvaala: Elementals toward next discount
    hero_free_roll_pending: bool = False  # Nozdormu: first refresh this turn is free
    hero_upgrade_discount: int = 0  # Chenvaala: accumulated next-upgrade discount
    pending_choice: Optional["PendingChoice"] = None
    placed_minion_board_index: Optional[int] = None
    placed_minion_pending_after: Optional["Minion"] = None
    triple_reward_discover_pending: bool = False
    triple_reward_spell_tier: int = 0
    # Signed normalized damage delta from each of the last ``BATTLE_HISTORY_LEN``
    # combats (most recent last). Empty until the player has fought at least once.
    battle_history: Tuple[float, ...] = ()
    # Snapshot of how many minions of each race were on this player's board at
    # the moment their last combat started (i.e. end-of-shop board). Drives the
    # "≥4 of a tribe" lock indicator. Empty dict until first combat; frozen at
    # elimination so dead opponents still expose their final composition.
    last_round_tribe_counts: Dict[Race, int] = field(default_factory=dict)
    # Snapshots of own + opp boards for this seat's most recent combat, in
    # own/opp orientation. Populated by ``resolve_combat_round``. The auxiliary
    # battle-prediction head consumes ``[0]`` (initial pre-combat snapshot);
    # future mid-battle snapshots will be appended.
    last_battle_snapshots: Tuple["BattleSnapshot", ...] = ()
    # Signed uncapped winner-damage from the most recent combat, signed from
    # this seat's perspective (+raw if won, -raw if lost, 0 if draw / no combat).
    last_battle_raw_signed: float = 0.0
    # True if this seat attacked first in the most recent combat.
    last_attack_first: bool = False

    @property
    def shopping_finished(self) -> bool:
        return self.phase == PlayerPhase.DONE
