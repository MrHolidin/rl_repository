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
    AttackOnKill,
    BuffCombatSummons,
    EveryNthTavernSpellFree,
    GoldNextTurnOnSell,
    GoldOnBuyTribe,
    OnNthDeathAddRaceToHand,
    OnNthSellAddRaceToHand,
    SummonCopyWhenSpace,
    FreeFirstRefreshEachTurn,
    GoldOnUpgrade,
    OnSellBuffRandomShop,
    OnSellRaceAddToShop,
    RotatingBuyTribeBuff,
    OnNthBuyAddCardToHand,
    PowerCostGrowsPerUse,
    OnTiersBoughtAddCardToHand,
    OnAttacksAddCardToHand,
    FreeBuyEachTurnAfterAttacks,
    ShopStatBuffPerBuys,
    TavernSpellBonusPerTurns,
    CastRandomSpellEachTurn,
    OnRefreshCopyHighestTier,
    OnRefreshGrantBonusKeyword,
    SkipTurnsThenDiscover,
    DiscoverAtTierOnGoldSpent,
    DiscoverHeroPowerOnTurn,
    FewerShopSlots,
    FreezeShopEachTurn,
    StartOfCombatBuffEnds,
    StartOfCombatBuffOnePerTribe,
    StartHandToken,
    StartTierMinions,
    UpgradeDiscountPerElementals,
    ZeroGoldForRounds,
)
from src.bg_core.minion import Minion, Race
from src.bg_lobby.player import PlayerState
from src.bg_recruitment.hand_slots import first_free_hand_slot
from src.bg_recruitment.shop import add_random_minion_to_hand, add_random_minion_to_shop

__all__ = [
    "assign_random_hero",
    "apply_hero_on_game_start",
    "apply_hero_on_turn_start",
    "apply_hero_after_shop_refresh",
    "can_use_hero_power",
    "hero_power_cost",
    "use_hero_power",
    "flush_hero_tier_discovers",
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


def _races_a_power_needs(hero) -> frozenset:
    """Every tribe the hero's *power* names.

    The power only, not the passives. A passive that cannot fire costs the seat
    nothing — Ysera's guaranteed slot quietly shows an ordinary minion — but a
    power charges gold for a Discover with nothing to fill it, which is a trap
    rather than a weak hero. It is also what keeps the 2021 pool, all fifteen
    of it passive, dealt exactly as it was.

    Read off the fields rather than listed per hero, so a new effect that names
    a tribe is covered by having named it.
    """
    found = set()
    sources = [ab.effect for ab in hero.power]
    for src in sources:
        for field in ("tribe", "race", "filter_race", "race_filter"):
            value = getattr(src, field, None)
            if isinstance(value, Race):
                found.add(value)
        for field in ("tribes", "races"):
            for value in getattr(src, field, ()) or ():
                if isinstance(value, Race):
                    found.add(value)
    return frozenset(found)


def assign_random_hero(
    player: PlayerState,
    *,
    patch: PatchContext,
    rng: np.random.Generator,
    shop_excluded_race=None,
) -> None:
    """Assign one random hero from the patch pool (deterministic given ``rng``).

    A hero whose power is a tribe this lobby left out is not offered:
    Alexstrasza with no Dragons pays a gold for a Discover that cannot be
    filled. A *passive* tied to an absent tribe is left alone — it costs the
    seat nothing, and the 2021 pool is passive to the last hero.
    """
    from src.bg_catalog.cards import normalize_shop_excluded_races

    pool = patch.hero_pool_ids
    if not pool:
        return
    out = frozenset(normalize_shop_excluded_races(shop_excluded_race) or ())
    eligible = sorted(pool)
    if out:
        served = [h for h in eligible if not (_races_a_power_needs(patch.heroes[h]) & out)]
        if served:
            eligible = served
    hid = eligible[int(rng.integers(0, len(eligible)))]
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
    if h.start_armor:
        player.armor = h.start_armor
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
        elif isinstance(p, DiscoverAtTierOnGoldSpent):
            # "Discover a Tier 7 minion to get after you spend 60 Gold" — the
            # pick is made now and held; the gold is what releases it.
            from .discover_pool import shop_pool_for_tier

            pool = sorted(
                shop_pool_for_tier(p.tier, shop_excluded_race=shop_excluded_race, patch=patch)
            )
            if pool:
                player.hero_promised_card = pool[int(rng.integers(0, len(pool)))]
        elif isinstance(p, ZeroGoldForRounds):
            pass
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
    # Recorded whether or not there is a hero: the mask asks the seat what
    # round it is playing, and a seat is not told otherwise.
    player.round_number = int(round_number)
    h = player.hero
    if h is None:
        return
    player.hero_power_uses_this_turn = 0
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
        elif isinstance(p, FreeBuyEachTurnAfterAttacks):
            # "the first minion you buy each turn is free", once the attacks
            # the card counts have happened.
            if player.hero_attacks >= p.attacks:
                player.hero_free_buys = 1
        elif isinstance(p, TavernSpellBonusPerTurns):
            # "improve this at the start of every 3 turns" — the bonus is the
            # seat's own standing one, raised on the turns the card names.
            if p.per_turns > 0 and int(round_number) % p.per_turns == 1:
                player.tavern_spell_bonus_attack += p.attack
                player.tavern_spell_bonus_health += p.health
        elif isinstance(p, SkipTurnsThenDiscover):
            if int(round_number) in p.rounds:
                player.gold = 0
            elif int(round_number) == max(p.rounds) + 1:
                player.hero_pending_tier_discovers = tuple(int(t) for t in p.tiers)
        elif isinstance(p, DiscoverHeroPowerOnTurn):
            due = p.every_turn or int(round_number) == int(p.on_turn)
            if due:
                _open_hero_power_discover(player, rng=rng, patch=patch, options=p.options)
    flush_hero_tier_discovers(
        player, rng=rng, patch=patch, shop_excluded_race=shop_excluded_race
    )
    _pay_gold_spent_heroes(player, rng=rng, patch=patch)
    apply_hero_on_attacks(player, patch=patch)
    apply_hero_on_deaths(
        player, rng=rng, patch=patch, shop_excluded_race=shop_excluded_race
    )


# --------------------------------------------------------------------------- #
# Buy / sell / upgrade / elemental
# --------------------------------------------------------------------------- #


def apply_hero_on_bought(
    minion: Minion,
    player: PlayerState,
    *,
    rng: Optional[np.random.Generator] = None,
    patch: Optional[PatchContext] = None,
) -> None:
    h = player.hero
    if h is None:
        return
    # What the buy was worth, counted once and read by however many passives
    # ask. A hero that pays per Tier and one that pays per Battlecry are two
    # questions about the same purchase — and the plain count of buys is a
    # third, read by a power that improves ("after you buy 4 cards") and by a
    # counter buff that grows, neither of which is a passive of its own.
    player.hero_buy_count += 1
    player.hero_tiers_bought += max(0, int(minion.tier))
    if _has_battlecry(minion):
        player.hero_battlecry_buys += 1
    for p in h.passives:
        if isinstance(p, EveryNthBuyBuff):
            if p.n > 0 and player.hero_buy_count % p.n == 0:
                minion.bonus_attack += p.attack
                minion.bonus_health += p.health
        elif isinstance(p, RotatingBuyTribeBuff):
            tribe = player.hero_rotating_tribe
            if tribe is not None and minion_matches_tribe(minion, tribe):
                minion.bonus_attack += p.attack
                minion.bonus_health += p.health
        elif isinstance(p, GoldOnBuyTribe):
            if minion_matches_tribe(minion, p.race):
                player.gold += p.amount
        elif isinstance(p, OnNthBuyAddCardToHand):
            if p.once and player.hero_once_paid:
                continue
            counted = (
                player.hero_battlecry_buys
                if p.require_battlecry
                else player.hero_buy_count
            )
            if p.n > 0 and counted > 0 and counted % p.n == 0:
                if _give_card(player, p.card_id, patch):
                    player.hero_once_paid = True
        elif isinstance(p, OnTiersBoughtAddCardToHand):
            if p.n <= 0:
                continue
            owed = player.hero_tiers_bought // p.n - player.hero_tiers_paid
            for _ in range(max(0, owed)):
                _give_card(player, p.card_id, patch)
                player.hero_tiers_paid += 1


def flush_hero_tier_discovers(
    player: PlayerState,
    *,
    rng: np.random.Generator,
    patch: PatchContext,
    shop_excluded_race: Optional[Race] = None,
    shared_pool=None,
) -> None:
    """Open the next Discover A.F. Kay or Faelin still owes, if the seat is idle.

    One at a time, because the tiers differ and the modal chain re-rolls the
    same kind — "Tiers 6, 4, and 2" is three different Discovers, not one
    repeated three times. Called at the same post-action moment the waiting
    Spellcraft spell is handed over.
    """
    from .tavern_spells import _open_tier_discover

    while player.hero_pending_tier_discovers and player.pending_choice is None:
        tier, *rest = player.hero_pending_tier_discovers
        _open_tier_discover(
            player,
            int(tier),
            rng=rng,
            patch=patch,
            shop_excluded_race=shop_excluded_race,
            shared_pool=shared_pool,
            repeats=1,
        )
        if player.pending_choice is None:
            # It refused — a full hand, or nothing left in the pool at that
            # Tier. The queue is what makes the chain open one at a time, so
            # taking the head off before knowing it opened threw the whole
            # chain away on one bad moment. It stays owed.
            break
        player.hero_pending_tier_discovers = tuple(rest)


def _open_hero_power_discover(player, *, rng, patch, options: int = 2) -> None:
    """Offer other heroes' powers, the seat's own left out."""
    from src.bg_lobby.player import PendingChoice, PendingChoiceKind

    held = player.hero.hero_id if player.hero is not None else None
    # A passive hero has no power to offer, and picking one would *delete* the
    # seat's rather than replace it — a third of the pool, before this.
    pool = sorted(
        cid
        for cid in patch.hero_pool_ids
        if cid != held and patch.heroes[cid].has_power()
    )
    if len(pool) < options:
        return
    picks = []
    remaining = list(pool)
    for _ in range(options):
        picks.append(remaining.pop(int(rng.integers(0, len(remaining)))))
    player.pending_choice = PendingChoice(
        PendingChoiceKind.HERO_POWER_DISCOVER, tuple(picks), 0
    )


def _pay_gold_spent_heroes(player, *, rng, patch) -> None:
    """Thorim: the minion picked at the start arrives once the gold is spent."""
    h = player.hero
    if h is None:
        return
    for p in h.passives:
        if not isinstance(p, DiscoverAtTierOnGoldSpent):
            continue
        if player.hero_gold_spent_total < p.gold or player.hero_gold_paid:
            continue
        held = player.hero_promised_card
        if held:
            if _give_card(player, held, patch):
                player.hero_gold_paid = 1
                player.hero_promised_card = ""


def _has_battlecry(minion: Minion) -> bool:
    from src.bg_core.effects import Trigger

    return any(ab.trigger is Trigger.ON_PLACE for ab in minion.abilities)


def _give_card(player: PlayerState, card_id: str, patch: Optional[PatchContext]) -> bool:
    """Put a named card in hand, minion or spell. False if it did not fit.

    A hero pays in cards this package already carries — a Brann, a Tavern
    Coin, a Triple Reward — so this looks the id up in both catalogs rather
    than assuming which kind it is.
    """
    if patch is None:
        return False
    slot = first_free_hand_slot(player)
    if slot is None:
        return False
    if card_id in patch.templates:
        player.hand[slot] = make_minion(card_id, patch=patch)
        return True
    spell = patch.tavern_spells.get(card_id)
    if spell is None:
        return False
    player.hand[slot] = spell
    return True


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
        elif isinstance(p, GoldNextTurnOnSell):
            # Next turn, not this one: the gold is banked and the turn start
            # pays it, which is the whole shape of the card.
            player.gold_next_turn += p.amount
        elif isinstance(p, OnNthSellAddRaceToHand):
            player.hero_sell_count += 1
            if p.n > 0 and player.hero_sell_count % p.n == 0:
                add_random_minion_to_hand(
                    player,
                    p.race,
                    shop_excluded_race,
                    rng=rng,
                    patch=patch,
                )
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


def hero_tavern_spell_is_free(player: PlayerState) -> bool:
    """Whether the *next* Tavern spell the seat buys costs nothing.

    Asked before the purchase and spent by it, the way every other "every Nth"
    promise on a seat is: the count moves when the card is bought, not when
    the price is quoted, or reading the price twice would move it twice.
    """
    h = player.hero
    if h is None:
        return False
    for p in h.passives:
        if isinstance(p, EveryNthTavernSpellFree) and p.n > 0:
            return (player.hero_tavern_spell_count + 1) % p.n == 0
    return False


def apply_hero_on_tavern_spell_bought(player: PlayerState) -> None:
    """Count one Tavern spell for the hero that pays every third."""
    h = player.hero
    if h is None:
        return
    if any(isinstance(p, EveryNthTavernSpellFree) for p in h.passives):
        player.hero_tavern_spell_count += 1


def apply_hero_on_level_up(player: PlayerState) -> None:
    h = player.hero
    if h is None:
        return
    for p in h.passives:
        if isinstance(p, GoldOnUpgrade):
            player.gold += p.amount


def apply_hero_on_refresh(player: PlayerState, *, rng: np.random.Generator) -> None:
    """What a hero does to the counter it has just rolled."""
    from copy import copy as _copy

    from src.bg_core.minion import BONUS_KEYWORDS

    h = player.hero
    if h is None:
        return
    for p in h.passives:
        filled = [i for i, m in enumerate(player.shop) if m is not None]
        if not filled:
            return
        if isinstance(p, OnRefreshCopyHighestTier):
            # "copy its highest-Tier minion and Freeze them both" — the copy
            # takes a free slot if there is one, and both are pinned so the
            # next roll leaves them alone.
            best = max(filled, key=lambda i: player.shop[i].tier)
            empty = next(
                (i for i in range(len(player.shop)) if player.shop[i] is None), None
            )
            if empty is None:
                continue
            player.shop[empty] = _copy(player.shop[best])
            frozen = list(player.shop_frozen)
            for slot in (best, empty):
                if slot < len(frozen):
                    frozen[slot] = True
            player.shop_frozen = tuple(frozen)
        elif isinstance(p, OnRefreshGrantBonusKeyword):
            keywords = sorted(BONUS_KEYWORDS, key=lambda k: k.name)
            for _ in range(max(1, int(p.repeats))):
                target = player.shop[filled[int(rng.integers(0, len(filled)))]]
                keyword = keywords[int(rng.integers(0, len(keywords)))]
                target.granted_keywords = target.granted_keywords | {keyword}
                if keyword.name == "SHIELD":
                    target.has_shield = True


def _improved(effect, level: int):
    """The power's numbers at the level the seat has bought its way to.

    "Improves after you buy 4 cards" multiplies what one use is worth rather
    than replacing it, so the card still prints the unit.
    """
    from dataclasses import fields as _fields, replace as _replace

    if level <= 1:
        return effect
    names = {f.name for f in _fields(effect)} & {"attack", "health", "amount"}
    changed = {
        name: int(getattr(effect, name)) * level
        for name in names
        if int(getattr(effect, name) or 0)
    }
    return _replace(effect, **changed) if changed else effect


def hero_power_cost(player: PlayerState) -> int:
    """What pressing it costs right now, the printed price plus what it has
    climbed to (Elise's "costs (1) more after each use")."""
    h = player.hero
    if h is None:
        return 0
    return max(0, int(h.power_cost) + int(player.hero_power_cost_delta))


def can_use_hero_power(player: PlayerState, round_number: int = 1) -> bool:
    """Whether the seat may press its power.

    Everything the card prints and nothing the engine invents: it has a power,
    it has woken up, it has a use left, and the seat can pay. The phase and the
    open-modal question belong to the caller, which is the mask.
    """
    h = player.hero
    if h is None or not h.has_power():
        return False
    if h.power_charges and player.hero_power_uses_game >= int(h.power_charges):
        return False
    if player.hero_power_uses_this_turn >= max(1, int(h.power_uses)):
        return False
    if int(round_number) < int(player.hero_power_ready_on_round):
        return False
    if h.power_unlocks_at_tier and player.tavern_tier < h.power_unlocks_at_tier:
        return False
    if h.power_unlocks_on_turn and int(round_number) < h.power_unlocks_on_turn:
        return False
    if player.gold < hero_power_cost(player):
        return False
    return _power_can_land(player)


def _power_can_land(player: PlayerState) -> bool:
    """Whether pressing it would do anything at all.

    The tavern greys the button out rather than taking the gold for nothing,
    and the mask should not offer a press that provably does nothing either: a
    swap against an empty counter, a Discover with a full hand, "destroy a
    friendly Undead" with no Undead on the board. Only what an effect certainly
    needs is listed — a power whose requirement is not here is always offered,
    so the ladder can be short and wrong-in-one-direction only.
    """
    from src.bg_core.board_helpers import minion_matches_tribe
    from src.bg_core.effects import (
        AddRandomMinionToHandEffect,
        AddRandomTavernSpellToHandEffect,
        BuffTargetByTierEffect,
        BuffTargetFriendlyBattlecry,
        DestroyFriendlyEffect,
        DiscoverTribeEffect,
        GainBloodGemsEffect,
        GrantTemporaryBuffEffect,
        MakeFriendlyGoldenEffect,
        RefreshWithLastOpponentEffect,
        ReplaceTavernCardEffect,
        RetriggerFriendlyAbilityEffect,
        SellFriendlyForStatsEffect,
        StealTavernMinionEffect,
        SwapAttackBetweenTwoEffect,
        SwapWithTavernMinionEffect,
    )
    from src.bg_recruitment.hand_slots import first_free_hand_slot
    from src.bg_recruitment.tavern_spells import _buff_helps, temp_buff_helps as _temp_buff_helps

    h = player.hero
    if h is None:
        return False
    board = player.board
    on_counter = [m for m in player.shop if m is not None]
    hand_room = first_free_hand_slot(player) is not None

    for ability in h.power:
        e = ability.effect
        if isinstance(e, StealTavernMinionEffect):
            if not on_counter or not hand_room:
                return False
        elif isinstance(
            e,
            (
                AddRandomMinionToHandEffect,
                AddRandomTavernSpellToHandEffect,
                DiscoverTribeEffect,
                # A Blood Gem is a card in hand like any other, and a hand with
                # no room loses it.
                GainBloodGemsEffect,
            ),
        ):
            if not hand_room:
                return False
        elif isinstance(e, DestroyFriendlyEffect):
            if not any(
                e.filter_race is None or minion_matches_tribe(m, e.filter_race)
                for m in board
            ):
                return False
            # Both halves or neither: the body is gone the moment it is
            # destroyed, so a payout with nowhere to land loses the trade.
            wants_hand = e.then is not None or e.get_copy or e.discover_tiers_below
            if wants_hand and not hand_room:
                return False
        elif isinstance(e, SwapWithTavernMinionEffect):
            if not on_counter:
                return False
            if not any(not (e.exclude_golden and m.is_golden) for m in board):
                return False
        elif isinstance(e, ReplaceTavernCardEffect):
            if not on_counter:
                return False
        elif isinstance(e, MakeFriendlyGoldenEffect):
            if e.in_tavern:
                if not on_counter:
                    return False
            elif not any(not m.is_golden for m in board):
                return False
        elif isinstance(e, RetriggerFriendlyAbilityEffect):
            if not any(
                any(ab.trigger == e.trigger for ab in m.abilities) for m in board
            ):
                return False
        elif isinstance(e, BuffTargetFriendlyBattlecry):
            picky = _buff_helps(e)
            if not any(
                (e.filter_race is None or minion_matches_tribe(m, e.filter_race))
                and (picky is None or picky(m))
                for m in board
            ):
                return False
        elif isinstance(e, (SellFriendlyForStatsEffect, SwapAttackBetweenTwoEffect)):
            # Two bodies: one to spend and one to receive.
            if len(board) < 2:
                return False
        elif isinstance(e, BuffTargetByTierEffect):
            if not board:
                return False
        elif isinstance(e, GrantTemporaryBuffEffect):
            # Reborn until next turn is worth nothing to a body that already
            # has it, the same way a second Divine Shield is.
            if not any(_temp_buff_helps(e, m) for m in board):
                return False
        elif isinstance(e, RefreshWithLastOpponentEffect):
            if not any(m is not None for m in player.last_opponent_board):
                return False
    return True


def use_hero_power(
    player: PlayerState,
    *,
    rng: np.random.Generator,
    patch: PatchContext,
    round_number: int = 1,
    shop_excluded_race: Optional[Race] = None,
    shared_pool=None,
) -> None:
    """Press it: pay, count the use, and resolve what it does.

    Resolved through the Tavern-spell path because a hero power is the same
    kind of thing -- an effect with no body behind it -- so everything a spell
    can already do, a power can, and the one loud dispatcher covers both.
    """
    from .economy import note_gold_spent
    from .tavern_spells import apply_tavern_spell_effect

    h = player.hero
    if h is None or not can_use_hero_power(player, round_number):
        raise ValueError("hero power is not available")
    cost = hero_power_cost(player)
    player.gold -= cost
    note_gold_spent(player, cost, patch=patch)
    player.hero_power_uses_this_turn += 1
    player.hero_power_uses_game += 1
    if h.power_cooldown_turns:
        player.hero_power_ready_on_round = int(round_number) + int(
            h.power_cooldown_turns
        )
    for p in h.passives:
        if isinstance(p, PowerCostGrowsPerUse):
            player.hero_power_cost_delta += int(p.amount)
    level = 1
    if h.power_improve_per_buys:
        level += player.hero_buy_count // int(h.power_improve_per_buys)
    for ability in h.power:
        apply_tavern_spell_effect(
            player,
            _improved(ability.effect, level),
            rng=rng,
            patch=patch,
            shop_excluded_race=shop_excluded_race,
            shared_pool=shared_pool,
        )


def apply_hero_after_shop_refresh(
    player: PlayerState,
    round_number: int,
    *,
    rng: np.random.Generator,
    patch: PatchContext,
    shop_excluded_race: Optional[Race] = None,
) -> None:
    """After the lobby has built the turn's tavern, not before it.

    Yogg-Saron casts a random Tavern spell every turn, and half of what a
    Tavern spell can do is to the counter — buff an offer, swap one, steal one.
    Cast at the turn-start hook it landed on the *previous* turn's shop, which
    the refresh then threw away, so the spell only ever counted when it happened
    to be one of the ones that does not touch the tavern.
    """
    h = player.hero
    if h is None:
        return
    from src.bg_core.effects import CastRandomTavernSpellEffect

    from .tavern_spells import apply_tavern_spell_effect

    for p in h.passives:
        if isinstance(p, CastRandomSpellEachTurn) and int(round_number) >= p.unlocks_on_turn:
            apply_tavern_spell_effect(
                player,
                CastRandomTavernSpellEffect(),
                rng=rng,
                patch=patch,
                shop_excluded_race=shop_excluded_race,
            )


def apply_hero_on_turn_end(player: PlayerState) -> None:
    """Sindragosa: the Tavern Freezes at the end of each turn."""
    h = player.hero
    if h is None:
        return
    if any(isinstance(p, FreezeShopEachTurn) for p in h.passives):
        player.shop_freeze_next_round = True


def apply_hero_on_gold_spent(
    player: PlayerState,
    amount: int,
    *,
    rng: Optional[np.random.Generator] = None,
    patch: Optional[PatchContext] = None,
) -> None:
    """Count gold leaving the seat, for the heroes that read a lifetime total.

    And pay out on the spot: "after you spend 60 Gold" is a threshold the
    purchase crosses, so the card arrives in the hand that just paid rather
    than at the following turn's start.
    """
    if player.hero is None or amount <= 0:
        return
    player.hero_gold_spent_total += int(amount)
    if patch is not None:
        _pay_gold_spent_heroes(
            player, rng=rng if rng is not None else np.random.default_rng(0), patch=patch
        )


def apply_hero_on_deaths(
    player: PlayerState,
    *,
    rng: np.random.Generator,
    patch: PatchContext,
    shop_excluded_race: Optional[Race] = None,
) -> None:
    """Pay the heroes that count friendly deaths, and remember what was paid.

    Read at the seat's own turn start rather than at the death: the deaths
    happen inside a fight, and a fight hands what it owes to the seat instead
    of reaching into its hand mid-combat.
    """
    from .game_counts import DEATHS

    h = player.hero
    if h is None:
        return
    died = int(player.game_counts.get(DEATHS, 0))
    for p in h.passives:
        if not isinstance(p, OnNthDeathAddRaceToHand) or p.n <= 0:
            continue
        owed = died // p.n - player.hero_deaths_paid
        for _ in range(max(0, owed)):
            add_random_minion_to_hand(
                player, p.race, shop_excluded_race, rng=rng, patch=patch
            )
            player.hero_deaths_paid += 1


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


def hero_start_combat_ends(player: PlayerState):
    """Illidan: what the end minions gain, and whether they swing at once."""
    h = player.hero
    if h is None:
        return None
    for p in h.passives:
        if isinstance(p, StartOfCombatBuffEnds):
            return p
    return None


def hero_start_combat_one_per_tribe(player: PlayerState):
    """Wagtoggle: stats for a friendly of each type, improved by gold spent."""
    h = player.hero
    if h is None:
        return None
    for p in h.passives:
        if isinstance(p, StartOfCombatBuffOnePerTribe):
            level = 1
            if p.per_gold > 0:
                level += player.hero_gold_spent_total // p.per_gold
            return (p.attack * level, p.health * level)
    return None


def hero_counts_attacks(player: PlayerState) -> bool:
    """Whether this seat's hero cares how many of its minions have attacked."""
    h = player.hero
    if h is None:
        return False
    return any(
        isinstance(p, (OnAttacksAddCardToHand, FreeBuyEachTurnAfterAttacks))
        for p in h.passives
    )


def apply_hero_on_attacks(
    player: PlayerState, *, patch: Optional[PatchContext] = None
) -> None:
    """Pay the heroes that count friendly attacks, at the seat's turn start.

    The attacks happen in a fight; a fight hands what it owes to the seat.
    """
    h = player.hero
    if h is None:
        return
    for p in h.passives:
        if not isinstance(p, OnAttacksAddCardToHand) or p.n <= 0:
            continue
        owed = player.hero_attacks // p.n - player.hero_attacks_paid
        for _ in range(max(0, owed)):
            _give_card(player, p.card_id, patch)
            player.hero_attacks_paid += 1


def hero_combat_summon_buff(player: PlayerState):
    """What a minion summoned mid-combat arrives with (Greybough).

    ``(attack, health, keywords)``, all zero and empty for a seat whose hero
    says nothing about it — which is every seat but one.
    """
    h = player.hero
    if h is None:
        return (0, 0, frozenset())
    for p in h.passives:
        if isinstance(p, BuffCombatSummons):
            return (int(p.attack), int(p.health), frozenset(p.keywords))
    return (0, 0, frozenset())


def hero_attack_on_kill(player: PlayerState) -> int:
    """Attack a friendly keeps for killing something (Rokara)."""
    h = player.hero
    if h is None:
        return 0
    for p in h.passives:
        if isinstance(p, AttackOnKill):
            return int(p.amount)
    return 0


def hero_space_summon(player: PlayerState, round_number: int):
    """Whether this seat copies its biggest minion into a free combat slot.

    ``None`` unless the hero says so and the turn it unlocks on has come;
    otherwise ``"health"`` or ``"attack"`` — which of the two it copies.
    """
    h = player.hero
    if h is None:
        return None
    for p in h.passives:
        if isinstance(p, SummonCopyWhenSpace):
            if int(round_number) < int(p.unlocks_on_turn):
                return None
            return "health" if p.by_health else "attack"
    return None


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
