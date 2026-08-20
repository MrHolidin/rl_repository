"""BG-style discover pools (tier-weighted) and Adapt option sets."""

from __future__ import annotations

from typing import Sequence, Dict, List, Mapping, Optional, Tuple

import numpy as np

from src.bg_catalog.cards import (
    normalize_shop_excluded_races,
    shop_minion_allowed_with_exclusion,
    shop_pool_for_tier,
    templates,
)
from src.bg_catalog.patch_context import PatchContext, require_patch
from src.bg_catalog.ruleset import DEFAULT_RULESET
from src.bg_core.effects import Ability, Keyword, SummonEffect, Trigger
from src.bg_core.minion import Minion, Race
from src.bg_lobby.shared_pool import SharedCardPool

# Keys for Gentle Megasaur–style Adapt (HS Journey to Un'Goro set).
ADAPT_KEYS_ALL: Tuple[str, ...] = (
    "adapt_volcanic_might",
    "adapt_crackling_shield",
    "adapt_flaming_claws",
    "adapt_living_spore",
    "adapt_lightning_speed",
    "adapt_razor_claws",
    "adapt_rocky_carapace",
    "adapt_rockshell_armadillo",
    "adapt_massive",
    "adapt_molten_blade",
)

assert len(ADAPT_KEYS_ALL) == 10


def draw_from_pool(
    rng: np.random.Generator,
    card_ids: Sequence[str],
    count: int,
    *,
    shared_pool: Optional[SharedCardPool] = None,
) -> List[str]:
    """Draw distinct cards the way the tavern does: by remaining copies.

    A Discover is "limited to the shared pool", and the pool holds more copies
    of the cheap minions than the expensive ones — 15 of each Tier 1 against 7
    of each Tier 6 — so a card's chance is how many of it are left, and the
    spread skews low on its own. It skews further as the lobby buys the popular
    cards out, which is the part no formula can imitate.

    ``roll_card_id`` has drawn the tavern's own offers this way all along; this
    is the same draw, for the places that were picking uniformly among distinct
    card ids instead. With no pool to ask, every card is equally likely.
    """
    pool = list(card_ids)
    picks: List[str] = []
    for _ in range(min(int(count), len(pool))):
        weights = None
        if shared_pool is not None:
            w = np.array(
                [max(0.0, float(shared_pool.remaining_copies(cid))) for cid in pool],
                dtype=np.float64,
            )
            if w.sum() > 0:
                weights = w / w.sum()
        j = int(rng.choice(len(pool), p=weights))
        picks.append(pool.pop(j))
    return picks


def tribe_discover_card_ids(
    tribe: Optional[Race],
    *,
    patch: PatchContext,
    require_deathrattle: bool = False,
    require_battlecry: bool = False,
) -> List[str]:
    """The cards a Discover may offer, before the tier cap.

    ``tribe`` is the common narrowing and the ability flags are the other one:
    "Discover a Deathrattle minion" reads a property of the card the same way
    "Discover a Beast" reads its race, so both are filters over the same list.
    """
    from src.bg_catalog.patch_catalog import load_tavern_minions
    from src.envs.minibg.summon_pool import record_has_battlecry, record_has_deathrattle

    ctx = require_patch(patch, where="discover_pool.tribe_discover_card_ids")
    tpl = templates(patch=ctx)
    effects = dict(ctx.effects)
    # The mechanics tags live on the catalog rows, not on the built templates.
    mechanics = (
        {r.id: frozenset(r.mechanics) for r in load_tavern_minions(ctx.patch_dir / "catalog.json")}
        if (require_deathrattle or require_battlecry)
        else {}
    )
    out: List[str] = []
    for cid, m in tpl.items():
        if m.is_token:
            continue
        if tribe is not None and m.race != tribe:
            continue
        tags = mechanics.get(cid, frozenset())
        if require_deathrattle and not record_has_deathrattle(cid, tags, effects):
            continue
        if require_battlecry and not record_has_battlecry(cid, tags, effects):
            continue
        out.append(cid)
    return out


def roll_discover_tribe_triple(
    rng: np.random.Generator,
    tavern_tier: int,
    shop_excluded_race: Optional[Race] = None,
    *,
    tribe: Optional[Race],
    shared_pool: Optional[SharedCardPool] = None,
    patch: PatchContext,
    require_deathrattle: bool = False,
    require_battlecry: bool = False,
    exact_tier: bool = False,
) -> Optional[Tuple[str, ...]]:
    """Up to three options of one tribe, at the seat's own tavern tier or below.

    Not one tier up: that is the Triple Reward's rule and only its rule. A
    Discover printed on a card reaches above the seat only when the card says
    so ("from a Tier higher"), and a bare "Discover a <tribe>" says nothing, so
    it takes the default.

    A tribe left out of the lobby offers nothing, which is the same answer the
    tavern gives: there are no Undead to Discover in a game without Undead.
    """
    ctx = require_patch(patch, where="discover_pool.roll_discover_tribe_triple")
    tpl = ctx.templates
    cap = min(ctx.meta.ruleset.max_tier, tavern_tier)
    if tribe is not None and tribe in normalize_shop_excluded_races(shop_excluded_race):
        eligible: List[str] = []
    else:
        eligible = [
            cid
            for cid in tribe_discover_card_ids(
                tribe,
                patch=ctx,
                require_deathrattle=require_deathrattle,
                require_battlecry=require_battlecry,
            )
            if (tpl[cid].tier == cap if exact_tier else tpl[cid].tier <= cap)
            and (
                tribe is not None
                or shop_minion_allowed_with_exclusion(tpl[cid], shop_excluded_race)
            )
        ]
    if shared_pool is not None:
        eligible = [cid for cid in eligible if shared_pool.remaining_copies(cid) > 0]
    if not eligible:
        return None
    # Fewer than three is a real outcome, not an error: capped at the seat's
    # own tier there are only two Tier-1 Beasts in the whole pool. The modal
    # offers what there is, and the legal mask offers that many picks.
    return tuple(draw_from_pool(rng, eligible, 3, shared_pool=shared_pool))


def roll_adapt_triple(rng: np.random.Generator) -> Tuple[str, str, str]:
    idx = rng.choice(len(ADAPT_KEYS_ALL), size=3, replace=False)
    keys = [ADAPT_KEYS_ALL[int(i)] for i in idx]
    return (keys[0], keys[1], keys[2])


def triple_reward_discover_tier(
    tavern_tier: int, *, patch: Optional[PatchContext] = None
) -> int:
    max_tier = patch.meta.ruleset.max_tier if patch is not None else DEFAULT_RULESET.max_tier
    return min(max_tier, int(tavern_tier) + 1)


def roll_triple_reward_discover_at_target_tier(
    rng: np.random.Generator,
    target_tier: int,
    shop_excluded_race: Optional[Race] = None,
    *,
    shared_pool: Optional[SharedCardPool] = None,
    patch: PatchContext,
) -> Optional[Tuple[str, str, str]]:
    ctx = require_patch(patch, where="discover_pool.roll_triple_reward_discover_at_target_tier")
    tpl = ctx.templates
    tgt = min(ctx.meta.ruleset.max_tier, max(1, int(target_tier)))
    eligible_exact = [
        cid
        for cid in shop_pool_for_tier(
            tgt, shop_excluded_race=shop_excluded_race, patch=ctx
        )
        if tpl[cid].tier == tgt
    ]
    eligible = eligible_exact
    if len(eligible) < 3:
        eligible = list(
            shop_pool_for_tier(tgt, shop_excluded_race=shop_excluded_race, patch=ctx)
        )
    if len(eligible) < 3:
        eligible = [
            cid
            for cid, m in tpl.items()
            if not m.is_token
            and not m.is_golden
            and m.tier <= tgt
            and shop_minion_allowed_with_exclusion(m, shop_excluded_race)
        ]
    if shared_pool is not None:
        eligible = [cid for cid in eligible if shared_pool.remaining_copies(cid) > 0]
    if len(eligible) < 3:
        if shared_pool is not None:
            return None
        raise RuntimeError(
            f"need at least 3 cards for triple-reward discover (tier {tgt}), got {len(eligible)}"
        )
    return tuple(draw_from_pool(rng, eligible, 3, shared_pool=shared_pool))


def roll_triple_reward_discover_triple(
    rng: np.random.Generator,
    tavern_tier: int,
    shop_excluded_race: Optional[Race] = None,
    *,
    shared_pool: Optional[SharedCardPool] = None,
    patch: PatchContext,
) -> Optional[Tuple[str, str, str]]:
    return roll_triple_reward_discover_at_target_tier(
        rng,
        triple_reward_discover_tier(tavern_tier, patch=patch),
        shop_excluded_race,
        shared_pool=shared_pool,
        patch=patch,
    )


def is_murloc_board_minion(m: Minion) -> bool:
    return m.race in (Race.MURLOC, Race.ALL)


def apply_adapt_key_to_minion(m: Minion, key: str) -> None:
    if key == "adapt_volcanic_might":
        m.bonus_attack += 1
        m.bonus_health += 1
    elif key == "adapt_crackling_shield":
        m.has_shield = True
        m.keywords = frozenset(m.keywords | {Keyword.SHIELD})
    elif key == "adapt_flaming_claws":
        m.bonus_attack += 3
    elif key == "adapt_living_spore":
        m.abilities = m.abilities + (
            Ability(
                Trigger.ON_DEATH,
                SummonEffect(token_id="adapt_plant", count=2),
            ),
        )
    elif key == "adapt_lightning_speed":
        m.keywords = frozenset(m.keywords | {Keyword.WINDFURY})
    elif key == "adapt_razor_claws":
        m.bonus_attack += 1
    elif key == "adapt_rocky_carapace":
        m.bonus_health += 3
    elif key == "adapt_rockshell_armadillo":
        m.bonus_attack += 1
        m.bonus_health += 3
        m.keywords = frozenset(m.keywords | {Keyword.TAUNT})
    elif key == "adapt_massive":
        m.bonus_attack += 3
        m.bonus_health += 3
    elif key == "adapt_molten_blade":
        m.bonus_attack += 1
        m.bonus_health += 2
    else:
        raise ValueError(f"unknown adapt key {key!r}")


__all__ = [
    "ADAPT_KEYS_ALL",
    "apply_adapt_key_to_minion",
    "is_murloc_board_minion",
    "tribe_discover_card_ids",
    "roll_adapt_triple",
    "roll_discover_tribe_triple",
    "roll_triple_reward_discover_at_target_tier",
    "roll_triple_reward_discover_triple",
    "triple_reward_discover_tier",
]
