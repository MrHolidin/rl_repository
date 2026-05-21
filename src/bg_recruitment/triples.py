"""Triple merge (forge golden) and triple-reward discover spell."""

from __future__ import annotations

from copy import copy
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.bg_catalog.card_pool import triple_merge_golden_abilities
from src.bg_catalog.cards import CARD_TEMPLATES
from src.bg_core.effects import Keyword
from src.bg_core.minion import Minion, Race
from src.bg_recruitment.discover_pool import (
    roll_triple_reward_discover_at_target_tier,
    triple_reward_discover_tier,
)
from .hand_slots import first_free_hand_slot
from src.bg_lobby.player import PendingChoice, PendingChoiceKind, PlayerState
from src.bg_lobby.shared_pool import SharedCardPool

from .pool_ledger import on_sell_minion

TRIPLE_REWARD_SPELL_CARD_ID = "triple_reward_discover"


def is_triple_reward_discover_spell(m: Minion) -> bool:
    return m.is_triple_reward_spell or m.card_id == TRIPLE_REWARD_SPELL_CARD_ID


def make_triple_reward_discover_spell(*, discover_tier: int) -> Minion:
    spell = copy(CARD_TEMPLATES[TRIPLE_REWARD_SPELL_CARD_ID])
    spell.triple_discover_tier = int(discover_tier)
    spell.is_triple_reward_spell = True
    return spell


def hand_has_free_slot(player: PlayerState) -> bool:
    return any(s is None for s in player.hand)


def merge_three_non_golden_into_golden(
    card_id: str, a: Minion, b: Minion, c: Minion
) -> Minion:
    tpl = CARD_TEMPLATES[card_id]
    merged_kw = (
        a.keywords
        | a.granted_keywords
        | b.keywords
        | b.granted_keywords
        | c.keywords
        | c.granted_keywords
    )
    shield = a.has_shield or b.has_shield or c.has_shield or (
        Keyword.SHIELD in merged_kw
    )
    return Minion(
        card_id=card_id,
        base_attack=tpl.base_attack * 2,
        base_health=tpl.base_health * 2,
        tier=tpl.tier,
        name=tpl.name,
        bonus_attack=a.bonus_attack + b.bonus_attack + c.bonus_attack,
        bonus_health=a.bonus_health + b.bonus_health + c.bonus_health,
        race=tpl.race,
        keywords=frozenset(merged_kw),
        granted_keywords=frozenset(),
        abilities=triple_merge_golden_abilities(card_id),
        has_shield=shield,
        is_token=tpl.is_token,
        is_golden=True,
        from_triple_merge=True,
        dbf_id=tpl.dbf_id,
    )


def grant_triple_reward_discover_spell(
    player: PlayerState,
    *,
    discover_tier: int,
) -> bool:
    """Put discover spell in hand. Returns False if no slot (caller sets pending)."""
    slot = first_free_hand_slot(player)
    if slot is None:
        return False
    player.hand[slot] = make_triple_reward_discover_spell(discover_tier=discover_tier)
    return True


def queue_triple_reward_discover_spell(
    player: PlayerState, *, discover_tier: int
) -> None:
    player.triple_reward_discover_pending = True
    player.triple_reward_spell_tier = int(discover_tier)


def resolve_one_triple(
    player: PlayerState,
    *,
    shared_pool: Optional[SharedCardPool] = None,
) -> bool:
    groups: Dict[str, List[Tuple[str, int, Minion]]] = {}
    for i, m in enumerate(player.board):
        if not m.is_golden and not is_triple_reward_discover_spell(m):
            groups.setdefault(m.card_id, []).append(("b", i, m))
    for i, hm in enumerate(player.hand):
        if (
            hm is not None
            and not hm.is_golden
            and not is_triple_reward_discover_spell(hm)
        ):
            groups.setdefault(hm.card_id, []).append(("h", i, hm))
    candidate: Optional[str] = None
    for cid in sorted(groups.keys()):
        if len(groups[cid]) >= 3:
            candidate = cid
            break
    if candidate is None:
        return False
    ordered = sorted(
        groups[candidate], key=lambda t: (0 if t[0] == "b" else 1, t[1])
    )[:3]
    m0, m1, m2 = ordered[0][2], ordered[1][2], ordered[2][2]
    if shared_pool is not None:
        for m in (m0, m1, m2):
            on_sell_minion(shared_pool, m)
    merged = merge_three_non_golden_into_golden(candidate, m0, m1, m2)
    if shared_pool is not None:
        if not shared_pool.acquire_new(merged.card_id, 3):
            raise RuntimeError(
                f"shared pool: cannot reserve 3 copies for golden {merged.card_id!r}"
            )
    for _, idx, _ in sorted((t for t in ordered if t[0] == "b"), key=lambda t: -t[1]):
        del player.board[idx]
    for _, idx, _ in sorted((t for t in ordered if t[0] == "h"), key=lambda t: -t[1]):
        player.hand[idx] = None
    hslot = first_free_hand_slot(player)
    assert hslot is not None, "triple merge with full hand (bug)"
    player.hand[hslot] = merged
    discover_tier = triple_reward_discover_tier(player.tavern_tier)
    if not grant_triple_reward_discover_spell(player, discover_tier=discover_tier):
        queue_triple_reward_discover_spell(player, discover_tier=discover_tier)
    return True


def resolve_triples_loop(
    player: PlayerState,
    *,
    shared_pool: Optional[SharedCardPool] = None,
) -> None:
    for _ in range(24):
        if not resolve_one_triple(player, shared_pool=shared_pool):
            break


def open_triple_reward_discover_modal(
    player: PlayerState,
    shop_excluded_race: Optional[Race],
    *,
    discover_tier: int,
    rng: np.random.Generator,
    shared_pool: Optional[SharedCardPool] = None,
) -> bool:
    from src.bg_recruitment.discover import try_open_hand_discover_modal

    opts = roll_triple_reward_discover_at_target_tier(
        rng,
        discover_tier,
        shop_excluded_race,
        shared_pool=shared_pool,
    )
    if opts is None:
        return False
    return try_open_hand_discover_modal(
        player,
        PendingChoiceKind.TRIPLE_REWARD_DISCOVER,
        opts,
        0,
        shared_pool=shared_pool,
    )


def play_triple_reward_discover_spell_from_hand(
    player: PlayerState,
    hand_slot: int,
    shop_excluded_race: Optional[Race],
    *,
    rng: np.random.Generator,
    shared_pool: Optional[SharedCardPool] = None,
) -> None:
    spell = player.hand[hand_slot]
    assert spell is not None and is_triple_reward_discover_spell(spell)
    tier = spell.triple_discover_tier or triple_reward_discover_tier(player.tavern_tier)
    player.hand[hand_slot] = None
    open_triple_reward_discover_modal(
        player,
        shop_excluded_race,
        discover_tier=tier,
        rng=rng,
        shared_pool=shared_pool,
    )


def flush_triple_reward_queue_if_idle(
    player: PlayerState,
    shop_excluded_race: Optional[Race],
    *,
    rng: np.random.Generator,
) -> None:
    if player.pending_choice is not None or not player.triple_reward_discover_pending:
        return
    tier = player.triple_reward_spell_tier or triple_reward_discover_tier(
        player.tavern_tier
    )
    if grant_triple_reward_discover_spell(player, discover_tier=tier):
        player.triple_reward_discover_pending = False
        player.triple_reward_spell_tier = 0


# Backward-compatible alias
try_open_triple_reward_discover = open_triple_reward_discover_modal

__all__ = [
    "TRIPLE_REWARD_SPELL_CARD_ID",
    "flush_triple_reward_queue_if_idle",
    "grant_triple_reward_discover_spell",
    "hand_has_free_slot",
    "is_triple_reward_discover_spell",
    "make_triple_reward_discover_spell",
    "merge_three_non_golden_into_golden",
    "open_triple_reward_discover_modal",
    "play_triple_reward_discover_spell_from_hand",
    "queue_triple_reward_discover_spell",
    "resolve_one_triple",
    "resolve_triples_loop",
    "try_open_triple_reward_discover",
]
