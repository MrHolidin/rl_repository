"""Triple merge (forge golden) and triple-reward discover queue."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.bg_catalog.card_pool import triple_merge_golden_abilities
from src.bg_catalog.cards import CARD_TEMPLATES
from src.bg_core.effects import Keyword
from src.bg_core.minion import Minion, Race
from src.envs.minibg.actions import HAND_SIZE
from src.bg_recruitment.discover_pool import roll_triple_reward_discover_triple
from src.envs.minibg.state import PendingChoice, PendingChoiceKind, PlayerState


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


def resolve_one_triple(player: PlayerState) -> bool:
    groups: Dict[str, List[Tuple[str, int, Minion]]] = {}
    for i, m in enumerate(player.board):
        if not m.is_golden:
            groups.setdefault(m.card_id, []).append(("b", i, m))
    for i, hm in enumerate(player.hand):
        if hm is not None and not hm.is_golden:
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
    merged = merge_three_non_golden_into_golden(candidate, m0, m1, m2)
    for _, idx, _ in sorted((t for t in ordered if t[0] == "b"), key=lambda t: -t[1]):
        del player.board[idx]
    for _, idx, _ in sorted((t for t in ordered if t[0] == "h"), key=lambda t: -t[1]):
        player.hand[idx] = None
    hslot = next((i for i in range(HAND_SIZE) if player.hand[i] is None), None)
    assert hslot is not None, "triple merge with full hand (bug)"
    player.hand[hslot] = merged
    return True


def resolve_triples_loop(player: PlayerState) -> None:
    for _ in range(24):
        if not resolve_one_triple(player):
            break


def try_open_triple_reward_discover(
    player: PlayerState,
    shop_excluded_race: Optional[Race],
    *,
    rng: np.random.Generator,
) -> None:
    if not hand_has_free_slot(player):
        player.triple_reward_discover_pending = True
        return
    player.triple_reward_discover_pending = False
    opts = roll_triple_reward_discover_triple(
        rng, player.tavern_tier, shop_excluded_race
    )
    player.pending_choice = PendingChoice(
        PendingChoiceKind.TRIPLE_REWARD_DISCOVER, opts, 0
    )


def flush_triple_reward_queue_if_idle(
    player: PlayerState,
    shop_excluded_race: Optional[Race],
    *,
    rng: np.random.Generator,
) -> None:
    if player.pending_choice is None and player.triple_reward_discover_pending:
        try_open_triple_reward_discover(player, shop_excluded_race, rng=rng)
