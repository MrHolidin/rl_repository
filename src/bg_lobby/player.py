"""Per-player recruitment state (shop phase); shared across rulesets."""

from __future__ import annotations

from copy import copy as _shallow_copy
from dataclasses import dataclass, field, fields as _dataclass_fields
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Union

from src.bg_catalog.ruleset import DEFAULT_RULESET, Ruleset
from src.bg_core.hero import Hero
from src.bg_core.minion import Minion, Race
from src.bg_core.spell_card import SpellCard
from src.envs.minibg.actions import MAX_SHOP_SLOTS

# What a hand slot can hold. `board`/`shop` stay Minion-only — only minions
# are ever placed or offered in the minion shop; a bought tavern spell sits
# in `hand` until PLAY consumes it (see src/bg_recruitment/triples.py).
HandCard = Union[Minion, SpellCard]

# History length for obs-side last-N-battles features. Length = 3 matches
# what real-BG shows on the opponent panel; bumping requires re-training.
BATTLE_HISTORY_LEN = 3

__all__ = [
    "BattleSnapshot",
    "HandCard",
    "Minion",
    "PlayerState",
    "PlayerPhase",
    "Race",
    "PendingChoiceKind",
    "PendingChoice",
    "CasterKind",
    "CasterRef",
    "apply_hero_damage",
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
    board: List[Minion]
    shop: List[Optional[Minion]]
    hand: List[Optional[HandCard]]
    phase: PlayerPhase
    shop_actions_used: int
    # Absorbs combat damage before health (modern per-hero balance lever via
    # ``Hero.start_armor``; 0 on classic/no-armor patches — see
    # ``apply_hero_damage``).
    armor: int = 0
    # The patch's numeric rules. Held here so the price of the next tier can be
    # derived rather than stored (see ``next_tier_up_cost`` below); shared and
    # immutable, so copying a player copies a reference, not a table.
    ruleset: Ruleset = DEFAULT_RULESET
    #: Rounds' worth of standing discount banked toward the next upgrade. Reset
    #: by the upgrade that spends it; the *price* is never written.
    upgrade_discount_accrued: int = 0
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
    # Cumulative count of minions this seat has BOUGHT, keyed by tribe. Pure
    # bookkeeping for the tribe-preference shaping: the trainer pays on the
    # delta, and a purchase is not otherwise recoverable from state (a bought
    # minion may be played, tripled or sold before anything reads the board).
    # Race.ALL is counted under ALL; a tribeless minion under None.
    bought_tribe_counts: Dict[Optional[Race], int] = field(default_factory=dict)
    # Per-seat tribe-preference vector, one component per tribe, drawn once at
    # game start. Read by the observation and by the shaping term; never
    # mutated during play.
    tribe_pref: Tuple[float, ...] = ()
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

    @property
    def next_tier_up_cost(self) -> int:
        """What the next tavern tier costs before one-shot levers and heroes.

        Derived, not stored. It used to be a field, which meant four places
        wrote it and two of them disagreed about where the base price came
        from: the package's table or the module-level default. On 19.6.0 that
        charged 11 for the step to tier 5 where the package says 9, and the
        number in meta.json never reached play. With the price computed there
        is no second source to disagree with, and nothing to keep in step.

        The floor at zero belongs here rather than in
        ``effective_level_up_cost``: the standing discount cannot go below
        free, but a hero surcharge on top of a free upgrade is still paid.
        """
        base = self.ruleset.level_up_cost(self.tavern_tier)
        return max(0, base - self.upgrade_discount_accrued)


def apply_hero_damage(player: PlayerState, damage: int) -> None:
    """Apply combat damage to ``player``, absorbing with ``armor`` first.

    Single choke point for hero-damage application (both the 2-player and
    8-player lobbies route through this) so armor — 0 and a no-op on patches
    that don't have it — behaves identically everywhere damage lands.
    """
    if damage <= 0:
        return
    absorbed = min(player.armor, damage)
    player.armor -= absorbed
    player.health -= damage - absorbed


# Fields whose value is a mutable container: a copy must clone them, or two
# states would share a list and mutating one would rewrite the other.
_CONTAINER_FIELDS = frozenset(
    {"board", "shop", "hand", "last_round_tribe_counts", "bought_tribe_counts"}
)

# Fields the copy rebuilds rather than carries: the pending-placement pair
# points AT board minions, so it has to be re-aimed at the clones.
_REMAPPED_FIELDS = frozenset(
    {"pending_choice", "placed_minion_board_index", "placed_minion_pending_after"}
)


def copy_player_state(p: PlayerState) -> PlayerState:
    """Value copy of a seat, carrying **every** field.

    Enumerated by ``dataclasses.fields`` rather than by hand. The hand-written
    version this replaces listed 22 of 38 fields, so the other 16 silently took
    their defaults on every copy -- and the copy runs once per *action*, not per
    turn. Measured on the live lobby: battle_history, last_opponent_board,
    shop_frozen, armor, last_combat_won and the economy counters survived
    exactly one decision after a combat and were zero for the rest of the turn.
    That reached play through Nomi, Steward of Time, Southsea Strongarm, Deck
    Swabbie, Refreshing Anomaly, Cap'n Hoggarr, Murozond and Hangry Dragon, and
    through the 13 battle-history floats in the observation.

    Nothing is reset here, deliberately. Every field that should be cleared
    already is, in the place that owns its lifetime: ``fire_on_turn_start``
    zeroes the per-turn counters, and ``economy`` clears the one-shot hero
    levers when they are spent. The old comment claiming this copy
    "intentionally resets" the transient economy fields was wrong -- it reset
    them a second time and at the wrong moment.

    Enumerating the fields also flips the direction of the next mistake: a field
    added to :class:`PlayerState` is now carried by default, and skipping it has
    to be written down.
    """
    new_board = [m.__copy__() for m in p.board]

    remapped_pending: Optional[Minion] = None
    pend = p.placed_minion_pending_after
    if pend is not None:
        try:
            i = p.board.index(pend)
        except ValueError:
            pass
        else:
            if 0 <= i < len(new_board):
                remapped_pending = new_board[i]
    placed_idx: Optional[int] = None
    if remapped_pending is not None:
        try:
            placed_idx = new_board.index(remapped_pending)
        except ValueError:
            placed_idx = None

    pc = p.pending_choice
    values = {
        "board": new_board,
        "shop": [m.__copy__() if m is not None else None for m in p.shop],
        # hand can hold a SpellCard (no custom __copy__) alongside Minion —
        # the generic copy.copy() dispatches correctly for both.
        "hand": [_shallow_copy(m) if m is not None else None for m in p.hand],
        "last_round_tribe_counts": dict(p.last_round_tribe_counts),
        "bought_tribe_counts": dict(p.bought_tribe_counts),
        "pending_choice": (
            PendingChoice(
                pc.kind,
                pc.options,
                pc.extra_modals_after,
                pc.options_pool_reserved,
                pc.transform_board_idx,
            )
            if pc is not None
            else None
        ),
        "placed_minion_board_index": placed_idx,
        "placed_minion_pending_after": remapped_pending,
    }
    for f in _dataclass_fields(PlayerState):
        if f.name not in values:
            values[f.name] = getattr(p, f.name)
    return PlayerState(**values)
