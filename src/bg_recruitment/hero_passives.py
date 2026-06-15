"""Apply hero passive powers at the recruitment/combat event sites.

Single dispatch hub for hero passives (patch 19.6 pool defined in
``data/bgcore/19_6_0_74257/heroes.py``). Every entry point is a no-op when the
seat has no hero, so the classic (no-hero) path is untouched.

Effects that need to persist across shop *actions* write only to dedicated
``hero_*`` fields on :class:`PlayerState` (carried by ``BGLikeGame._copy_player``)
or to fields the copy already preserves (gold, hand, shop, board). Costs that
change every read (Millhouse flat costs, Millificent/Ysera shop generation,
Deathwing/Al'Akir combat) are derived from ``player.hero`` at the use site.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.bg_catalog.cards import (
    make_minion,
    shop_minion_allowed_with_exclusion,
)
from src.bg_catalog.patch_context import PatchContext
from src.bg_core.board_helpers import minion_matches_tribe
from src.bg_core.hero import (
    EveryNthBuyBuff,
    FreeFirstRefreshEachTurn,
    GoldOnUpgrade,
    OnSellBuffRandomShop,
    OnSellRaceAddToShop,
    RotatingBuyTribeBuff,
    StartHandToken,
    StartTierMinions,
    UpgradeDiscountPerElementals,
    ZeroGoldForRounds,
)
from src.bg_core.minion import Minion, Race
from src.bg_lobby.player import PlayerState
from src.bg_recruitment.hand_slots import first_free_hand_slot
from src.bg_recruitment.shop import add_random_minion_to_shop

__all__ = [
    "assign_random_hero",
    "apply_hero_on_game_start",
    "apply_hero_on_turn_start",
    "apply_hero_on_bought",
    "apply_hero_on_sell",
    "apply_hero_on_level_up",
    "apply_hero_on_elemental_played",
    "hero_combat_attack_aura",
    "hero_start_combat_keywords",
]


# --------------------------------------------------------------------------- #
# Assignment
# --------------------------------------------------------------------------- #


def assign_random_hero(
    player: PlayerState,
    *,
    patch: PatchContext,
    rng: np.random.Generator,
) -> None:
    """Assign one random hero from the patch pool (deterministic given ``rng``)."""
    pool = patch.hero_pool_ids
    if not pool:
        return
    hid = sorted(pool)[int(rng.integers(0, len(pool)))]
    player.hero = patch.heroes[hid]


# --------------------------------------------------------------------------- #
# Game start / turn start
# --------------------------------------------------------------------------- #


def apply_hero_on_game_start(
    player: PlayerState,
    round_number: int,
    *,
    patch: PatchContext,
    rng: np.random.Generator,
    shared_pool=None,
    shop_excluded_race: Optional[Race] = None,
) -> None:
    h = player.hero
    if h is None:
        return
    if h.start_health is not None:
        player.health = h.start_health
    for p in h.passives:
        if isinstance(p, StartHandToken):
            slot = first_free_hand_slot(player)
            if slot is not None:
                player.hand[slot] = make_minion(p.card_id, patch=patch)
        elif isinstance(p, StartTierMinions):
            _add_tier_minions_to_hand(
                player,
                p.count,
                p.tier,
                shop_excluded_race,
                rng=rng,
                shared_pool=shared_pool,
                patch=patch,
            )
    # Round-1 turn-start levers (Nozdormu free roll, Rat King initial tribe,
    # A.F. Kay gold 0). Subsequent rounds go through apply_hero_on_turn_start.
    apply_hero_on_turn_start(
        player, round_number, patch=patch, rng=rng, shop_excluded_race=shop_excluded_race
    )


def apply_hero_on_turn_start(
    player: PlayerState,
    round_number: int,
    *,
    patch: PatchContext,
    rng: np.random.Generator,
    shop_excluded_race: Optional[Race] = None,
) -> None:
    h = player.hero
    if h is None:
        return
    for p in h.passives:
        if isinstance(p, FreeFirstRefreshEachTurn):
            player.hero_free_roll_pending = True
        elif isinstance(p, RotatingBuyTribeBuff):
            player.hero_rotating_tribe = _roll_next_tribe(
                player.hero_rotating_tribe, patch, rng, shop_excluded_race
            )
        elif isinstance(p, ZeroGoldForRounds):
            if int(round_number) in p.rounds:
                player.gold = 0


# --------------------------------------------------------------------------- #
# Buy / sell / upgrade / elemental
# --------------------------------------------------------------------------- #


def apply_hero_on_bought(minion: Minion, player: PlayerState) -> None:
    h = player.hero
    if h is None:
        return
    for p in h.passives:
        if isinstance(p, EveryNthBuyBuff):
            player.hero_buy_count += 1
            if p.n > 0 and player.hero_buy_count % p.n == 0:
                minion.bonus_attack += p.attack
                minion.bonus_health += p.health
        elif isinstance(p, RotatingBuyTribeBuff):
            tribe = player.hero_rotating_tribe
            if tribe is not None and minion_matches_tribe(minion, tribe):
                minion.bonus_attack += p.attack
                minion.bonus_health += p.health


def apply_hero_on_sell(
    sold: Minion,
    player: PlayerState,
    *,
    rng: np.random.Generator,
    patch: PatchContext,
    shared_pool=None,
    shop_excluded_race: Optional[Race] = None,
) -> None:
    h = player.hero
    if h is None:
        return
    for p in h.passives:
        if isinstance(p, OnSellBuffRandomShop):
            _buff_random_shop(player, p.count, p.attack, p.health, rng)
        elif isinstance(p, OnSellRaceAddToShop):
            if minion_matches_tribe(sold, p.race):
                add_random_minion_to_shop(
                    player,
                    p.race,
                    shop_excluded_race,
                    rng=rng,
                    shared_pool=shared_pool,
                    patch=patch,
                )


def apply_hero_on_level_up(player: PlayerState) -> None:
    h = player.hero
    if h is None:
        return
    for p in h.passives:
        if isinstance(p, GoldOnUpgrade):
            player.gold += p.amount


def apply_hero_on_elemental_played(player: PlayerState) -> None:
    h = player.hero
    if h is None:
        return
    for p in h.passives:
        if isinstance(p, UpgradeDiscountPerElementals):
            player.hero_elementals_progress += 1
            if p.per > 0 and player.hero_elementals_progress >= p.per:
                player.hero_elementals_progress -= p.per
                player.hero_upgrade_discount += p.reduction


# --------------------------------------------------------------------------- #
# Combat (read by eight_player → simulate_battle)
# --------------------------------------------------------------------------- #


def hero_combat_attack_aura(player: PlayerState) -> int:
    h = player.hero
    return h.combat_attack_aura() if h is not None else 0


def hero_start_combat_keywords(player: PlayerState) -> frozenset:
    h = player.hero
    return h.start_combat_leftmost_keywords() if h is not None else frozenset()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _roll_next_tribe(
    current: Optional[Race],
    patch: PatchContext,
    rng: np.random.Generator,
    shop_excluded_race: Optional[Race],
) -> Optional[Race]:
    """Pick a rotation tribe != ``current`` (Rat King 'not twice in a row'),
    avoiding the round's excluded tribe(s) when possible."""
    tribes = list(patch.meta.rotation_tribes)
    if not tribes:
        return current
    excl: set = set()
    if shop_excluded_race is not None:
        if isinstance(shop_excluded_race, (tuple, list, set, frozenset)):
            excl.update(shop_excluded_race)
        else:
            excl.add(shop_excluded_race)
    cands = [t for t in tribes if t != current and t not in excl]
    if not cands:
        cands = [t for t in tribes if t != current] or tribes
    return cands[int(rng.integers(0, len(cands)))]


def _buff_random_shop(
    player: PlayerState, count: int, attack: int, health: int, rng: np.random.Generator
) -> None:
    idxs = [i for i, m in enumerate(player.shop) if m is not None]
    if not idxs:
        return
    for _ in range(max(0, count)):
        i = idxs[int(rng.integers(0, len(idxs)))]
        player.shop[i].bonus_attack += attack
        player.shop[i].bonus_health += health


def _add_tier_minions_to_hand(
    player: PlayerState,
    count: int,
    tier: int,
    shop_excluded_race: Optional[Race],
    *,
    rng: np.random.Generator,
    shared_pool,
    patch: PatchContext,
) -> None:
    tpl = patch.templates

    def candidates(respect_exclusion: bool):
        return [
            cid
            for cid, t in tpl.items()
            if not t.is_token
            and not t.is_golden
            and not t.is_triple_reward_spell
            and t.tier == tier
            and (
                shop_minion_allowed_with_exclusion(t, shop_excluded_race)
                if respect_exclusion
                else True
            )
        ]

    cands = candidates(True) or candidates(False)
    if not cands:
        return
    for _ in range(max(0, count)):
        slot = first_free_hand_slot(player)
        if slot is None:
            break
        cid = cands[int(rng.integers(0, len(cands)))]
        if shared_pool is not None:
            shared_pool.acquire_new(cid)  # best-effort pool accounting
        player.hand[slot] = make_minion(cid, patch=patch)
