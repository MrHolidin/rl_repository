from __future__ import annotations

from copy import copy
from typing import Dict, List

from .effects import (
    Ability,
    BuffRandomFriendly,
    Keyword,
    StatAura,
    SummonEffect,
    Trigger,
)
from .state import Minion


CARD_TEMPLATES: Dict[str, Minion] = {
    "recruit": Minion(
        card_id="recruit",
        base_attack=2,
        base_health=2,
        tier=1,
    ),
    "guard": Minion(
        card_id="guard",
        base_attack=1,
        base_health=3,
        tier=1,
        keywords=frozenset({Keyword.TAUNT}),
    ),
    "buffer": Minion(
        card_id="buffer",
        base_attack=1,
        base_health=1,
        tier=1,
        abilities=(
            Ability(Trigger.ON_BUY, BuffRandomFriendly(attack=1, health=1, exclude_self=True)),
        ),
    ),
    "bruiser": Minion(
        card_id="bruiser",
        base_attack=4,
        base_health=3,
        tier=2,
    ),
    "shield_bot": Minion(
        card_id="shield_bot",
        base_attack=2,
        base_health=2,
        tier=2,
        keywords=frozenset({Keyword.SHIELD}),
    ),
    "pack_rat": Minion(
        card_id="pack_rat",
        base_attack=2,
        base_health=2,
        tier=2,
        abilities=(Ability(Trigger.ON_DEATH, SummonEffect(token_id="rat_token")),),
    ),
    "big_guy": Minion(
        card_id="big_guy",
        base_attack=5,
        base_health=5,
        tier=3,
    ),
    "commander": Minion(
        card_id="commander",
        base_attack=3,
        base_health=4,
        tier=3,
        abilities=(Ability(Trigger.AURA, StatAura(attack=1)),),
    ),
    "summoner": Minion(
        card_id="summoner",
        base_attack=4,
        base_health=3,
        tier=3,
        abilities=(Ability(Trigger.ON_DEATH, SummonEffect(token_id="summoned_token")),),
    ),
    "rat_token": Minion(
        card_id="rat_token",
        base_attack=1,
        base_health=1,
        tier=1,
        is_token=True,
    ),
    "summoned_token": Minion(
        card_id="summoned_token",
        base_attack=2,
        base_health=2,
        tier=1,
        is_token=True,
    ),
}


def make_minion(card_id: str) -> Minion:
    template = CARD_TEMPLATES[card_id]
    fresh = copy(template)
    fresh.has_shield = Keyword.SHIELD in template.keywords
    return fresh


def shop_pool_for_tier(tavern_tier: int) -> List[str]:
    return [
        cid
        for cid, m in CARD_TEMPLATES.items()
        if not m.is_token and m.tier <= tavern_tier
    ]


__all__ = ["CARD_TEMPLATES", "make_minion", "shop_pool_for_tier"]
