"""Tavern spells — the cards Bob sells that are not minions.

Three things happen to one, and they are separate moves: the tavern **offers**
it, the seat **buys** it into hand, and the seat **plays** it. Buying is not
playing (a bought spell can sit in hand across the turn), and playing is not
buying (a spell can reach hand without ever being on the counter).

Where the offer lives is the one design choice here. A shop slot is a
``Minion`` slot to everything that reads one — the observation, the legal mask,
the flat buy actions — so a ``SpellCard`` dropped into ``player.shop`` would be
read as a minion by all three. The offer therefore sits in its own field. It
does not cost a minion slot: a tier-1 tavern shows three minions *and* a spell,
so the seat sees one more card than it used to, not the same number.

Like Blood Gems and Spellcraft before it, this is engine API only: the flat RL
action space has no "buy the spell" or "play a spell at a target" index, and
adding one would move every number a trained checkpoint is wired to.
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.bg_catalog.cards import normalize_shop_excluded_races
from src.bg_catalog.patch_context import PatchContext, require_patch
from src.bg_core.effects import (
    AddRandomTavernSpellToHandEffect,
    BuffAllShopOffersEffect,
    BuffSharedTribeEffect,
    CastSpellAtEffect,
    BuffTargetFriendlyBattlecry,
    CastRandomTavernSpellEffect,
    ChooseOneEffect,
    CopyLastTavernSpellEffect,
    DiscoverMinionAtTierEffect,
    DiscoverTavernSpellEffect,
    IncreaseTavernSpellBonusEffect,
    MakeFriendlyGoldenEffect,
    BloodGemsOnEveryRefreshEffect,
    DestroyFriendlyEffect,
    DiscoverHeroPowerEffect,
    MultiplierKind,
    PayInHealthEffect,
    PromiseNextTurnEffect,
    RefreshWithTavernSpellsEffect,
    RefreshWithTribeEffect,
    StealNeighbourBloodGemsEffect,
    SummonOnCombatSpaceEffect,
    RaiseStandingBonusEffect,
    SellFriendlyForStatsEffect,
    TransformToHigherTierEffect,
    StealTavernMinionEffect,
    Trigger,
)
from src.bg_core.board_helpers import (
    fire_spell_cast_on,
    minion_matches_tribe,
    multiplier_for,
)
from src.bg_core.minion import Minion, Race, next_instance_id
from src.bg_core.spell_card import SpellCard
from src.bg_lobby.player import PlayerPhase, PlayerState
from src.bg_lobby.shared_pool import SharedCardPool

from .game_counts import (
    SPELLS_CAST,
    TAVERN_SPELLS_CAST,
    bump_seat_counter,
    improve_level,
)
from .hand_slots import first_free_hand_slot
from .standing_bonuses import BonusScope, raise_standing_bonus
from .pool_ledger import on_bought_from_shop

__all__ = [
    "TavernSpellNotAllowed",
    "tavern_spell_pool",
    "effective_tavern_spell_cost",
    "offer_tavern_spells",
    "clear_tavern_spell_offers",
    "buy_tavern_spell",
    "spell_costs_health",
    "steal_tavern_minion",
    "add_random_tavern_spells",
    "apply_tavern_spell_effect",
    "cast_tavern_spell",
    "open_tavern_spell_discover",
    "spell_gives_stats",
    "tavern_spell_bonus",
    "play_tavern_spell_from_hand",
]


class TavernSpellNotAllowed(ValueError):
    """The seat cannot do this with a Tavern spell right now, and why."""


def tavern_spell_pool(
    tavern_tier: int,
    *,
    patch: PatchContext,
    shop_excluded_race=None,
) -> List[str]:
    """Spell ids the tavern can offer a seat at ``tavern_tier``.

    Same rule as the minion counter: everything up to the seat's tier, so a
    tier-1 spell keeps showing up all game — and, like the counter, nothing
    belonging to a tribe this lobby left out. The Bounties are Pirate-lobby
    spells, Spitescale Special a Naga one, Temperature Shift an Elemental one,
    and Temperature Shift is the reason this matters beyond flavour: it hands
    over two Elementals, so offering it in a lobby without Elementals put a
    tribe on the board that the rotation had excluded.
    """
    ctx = require_patch(patch, where="tavern_spells.tavern_spell_pool")
    excluded = set(normalize_shop_excluded_races(shop_excluded_race))
    return sorted(
        card_id
        for card_id, spell in ctx.tavern_spells.items()
        if spell.in_pool
        and 1 <= spell.tier <= int(tavern_tier)
        and ctx.spell_tribe_gates.get(card_id) not in excluded
    )


def spell_gives_stats(spell: SpellCard) -> bool:
    """Whether this spell hands out stats, asked of its bindings not its text.

    "Get a random Tavern spell **that gives stats**" is printed on three cards,
    and the catalog has no such flag — but a spell that gives stats is one whose
    effects buff something, which the bindings already say.
    """
    stat_giving = (BuffTargetFriendlyBattlecry, BuffAllShopOffersEffect)

    def _gives(effect) -> bool:
        if isinstance(effect, ChooseOneEffect):
            return _gives(effect.first) or _gives(effect.second)
        return isinstance(effect, stat_giving) and bool(effect.attack or effect.health)

    return any(_gives(ability.effect) for ability in spell.abilities)


def tavern_spell_bonus(player: PlayerState) -> Tuple[int, int]:
    """Extra stats this seat's Tavern spells hand out beyond the printed ones.

    Two sources, added: what the seat has banked "this game" (Intrepid Botanist
    and its kin, which keep paying after the card is gone) and what a body
    standing on the board says right now (Humon'gozz, which stops the moment it
    is sold — an aura, not a promise).
    """
    attack = int(player.tavern_spell_bonus_attack)
    health = int(player.tavern_spell_bonus_health)
    for minion in player.board:
        for ab in minion.abilities:
            if ab.trigger is Trigger.AURA and isinstance(
                ab.effect, IncreaseTavernSpellBonusEffect
            ):
                attack += int(ab.effect.attack)
                health += int(ab.effect.health)
    return (attack, health)


def effective_tavern_spell_cost(player: PlayerState, spell: SpellCard) -> int:
    """What this seat pays for ``spell`` — printed cost, discounts applied."""
    return max(0, int(spell.cost) + int(player.tavern_spell_cost_delta))


def clear_tavern_spell_offers(player: PlayerState) -> None:
    """Take every spell off the counter (a new tavern, or they were bought)."""
    player.tavern_spell_offers = ()


def offer_tavern_spells(
    player: PlayerState,
    *,
    rng: np.random.Generator,
    patch: PatchContext,
    card_ids: Optional[Sequence[str]] = None,
    count: Optional[int] = None,
    shop_excluded_race=None,
) -> Tuple[SpellCard, ...]:
    """Put this tavern's Tavern spells on the counter.

    How many is ``ruleset.tavern_spells_per_roll``; which ones is a draw from
    everything up to the seat's tier, without repeats. Nothing is displaced: the
    spell sits beside the minion row, which keeps this out of ``player.shop``
    and out of the shared-pool ledger entirely.

    ``card_ids`` names them instead of rolling, which is what tooling and tests
    want. Returns what was offered — empty on every package that carries no
    Tavern spells, which is every 2021 one.
    """
    ctx = require_patch(patch, where="tavern_spells.offer_tavern_spells")
    if card_ids is None:
        want = max(
            0,
            int(player.ruleset.tavern_spells_per_roll if count is None else count),
        )
        pool = tavern_spell_pool(
            player.tavern_tier, patch=ctx, shop_excluded_race=shop_excluded_race
        )
        picks: List[str] = []
        for _ in range(min(want, len(pool))):
            picks.append(pool.pop(int(rng.integers(0, len(pool)))))
        card_ids = picks

    offers: List[SpellCard] = []
    for card_id in card_ids:
        spell = ctx.tavern_spells.get(card_id)
        if spell is None or not spell.is_tavern_spell:
            raise TavernSpellNotAllowed(
                f"{card_id} is not a Tavern spell in this package"
            )
        offers.append(spell)
    player.tavern_spell_offers = tuple(offers)
    return player.tavern_spell_offers


def _note_spend(player: PlayerState, cost: int, *, patch) -> None:
    """Gold on a Tavern spell is gold spent, and the watchers want to know."""
    from .economy import note_gold_spent

    note_gold_spent(player, int(cost), patch=patch)


def spell_costs_health(spell: SpellCard) -> bool:
    """Whether this card is bought with Health instead of Gold.

    A property of the price rather than something that fires, so it is asked of
    the card at the counter and carried as an AURA marker on it.
    """
    return any(
        ability.trigger is Trigger.AURA
        and isinstance(ability.effect, PayInHealthEffect)
        for ability in spell.abilities
    )


def buy_tavern_spell(
    player: PlayerState,
    offer_index: int = 0,
    *,
    patch: PatchContext,
) -> SpellCard:
    """Pay for the spell at ``offer_index`` on the counter and put it in hand.

    Refuses loudly rather than doing nothing, the way ``activate_minion`` does:
    a purchase that silently failed looks exactly like one whose effect is not
    implemented.
    """
    require_patch(patch, where="tavern_spells.buy_tavern_spell")
    offers = player.tavern_spell_offers
    if not 0 <= offer_index < len(offers):
        raise TavernSpellNotAllowed(
            f"no Tavern spell on the counter at index {offer_index}"
        )
    spell = offers[offer_index]
    if player.phase != PlayerPhase.SHOP:
        raise TavernSpellNotAllowed("buying is a recruit-phase move")
    cost = effective_tavern_spell_cost(player, spell)
    in_health = spell_costs_health(spell)
    if not in_health and player.gold < cost:
        raise TavernSpellNotAllowed(
            f"{spell.card_id} costs {cost}; the seat has {player.gold}"
        )
    slot = first_free_hand_slot(player)
    if slot is None:
        raise TavernSpellNotAllowed("hand is full")

    if in_health:
        # One Health per Gold, and paid as hero damage so that armor absorbs it
        # and everything reading hero damage sees it — the same route the
        # refreshes-cost-Health cards take.
        from src.bg_lobby.player import apply_hero_damage

        apply_hero_damage(player, cost, patch=patch)
    else:
        player.gold -= cost
        _note_spend(player, cost, patch=patch)
    # The discount was for this purchase and is spent by it, whether or not it
    # was worth anything (a 0-cost spell still consumes Ominous Seer's promise).
    player.tavern_spell_cost_delta = 0
    player.hand[slot] = spell
    player.tavern_spell_offers = tuple(
        s for i, s in enumerate(offers) if i != offer_index
    )
    return spell


def play_tavern_spell_from_hand(
    player: PlayerState,
    hand_index: int,
    *,
    rng: np.random.Generator,
    patch: PatchContext,
    target_board_index: Optional[int] = None,
    target_shop_index: Optional[int] = None,
    choose_one_option: int = 0,
    shop_excluded_race: Optional[Race] = None,
    shared_pool: Optional[SharedCardPool] = None,
) -> None:
    """Cast the Tavern spell in ``hand_index``, then discard it.

    ``target_board_index`` is the friendly a "give a minion +X/+Y" names and
    ``target_shop_index`` is the same naming a minion still on the counter —
    the card says "a minion", not "a friendly minion", and buffing one before
    you buy it is an ordinary play. ``choose_one_option`` picks the half of a
    Choose One. All three are the seat's decisions; with none given the effect
    falls back to a random legal target, which is what the placement path
    already does.
    """
    ctx = require_patch(patch, where="tavern_spells.play_tavern_spell_from_hand")
    card = player.hand[hand_index] if 0 <= hand_index < len(player.hand) else None
    if not isinstance(card, SpellCard) or not card.is_tavern_spell:
        raise TavernSpellNotAllowed(f"hand slot {hand_index} holds no Tavern spell")
    if player.phase != PlayerPhase.SHOP:
        raise TavernSpellNotAllowed("casting is a recruit-phase move")

    target = None
    if target_board_index is not None and 0 <= target_board_index < len(player.board):
        target = player.board[target_board_index]
    elif target_shop_index is not None and 0 <= target_shop_index < len(player.shop):
        target = player.shop[target_shop_index]
    player.hand[hand_index] = None
    cast_tavern_spell(
        player,
        card,
        rng=rng,
        patch=ctx,
        target=target,
        choose_one_option=choose_one_option,
        shop_excluded_race=shop_excluded_race,
        shared_pool=shared_pool,
    )


def cast_tavern_spell(
    player: PlayerState,
    card: SpellCard,
    *,
    rng: np.random.Generator,
    patch: PatchContext,
    target: Optional[Minion] = None,
    choose_one_option: int = 0,
    shop_excluded_race: Optional[Race] = None,
    shared_pool: Optional[SharedCardPool] = None,
) -> None:
    """Resolve a Tavern spell that is already out of hand (or never was).

    Split from ``play_tavern_spell_from_hand`` because two cards cast a spell
    the seat never held: "Cast a random Tavern spell" makes one on the spot.
    Everything that follows a cast — the seat's spell bonus, the memory of
    which one it was, the listeners — belongs to the cast, not to the hand.
    """
    # "Your Bounties cast twice", "your spells that target friendly minions
    # cast twice" — the card is unchanged and the cast is what repeats, so the
    # whole resolution runs again rather than the numbers doubling. Both read
    # at once, and they compose: a Bounty aimed at a friendly under Balinda
    # casts four times.
    times = 1
    if card.card_id in patch.bounty_ids:
        times = max(1, multiplier_for(player.board, MultiplierKind.BOUNTY))
    if target is not None and target in player.board:
        # "Your spells that **target friendly minions** cast twice" — a minion
        # on the counter is not friendly, and buying it later does not make the
        # spell retroactively doubled.
        times *= max(1, multiplier_for(player.board, MultiplierKind.TARGETED_SPELL))
    for _ in range(times):
        _resolve_spell_abilities(
            player,
            card,
            rng=rng,
            patch=patch,
            target=target,
            choose_one_option=choose_one_option,
            shop_excluded_race=shop_excluded_race,
            shared_pool=shared_pool,
        )
    if target is not None:
        # A Tavern spell aimed at a body is a spell cast on it, the same event a
        # Blood Gem and a Spellcraft spell are. Only those two used to reach
        # here, so every "whenever you cast a spell on a <tribe>" card was blind
        # to the most ordinary way of doing it.
        fire_spell_cast_on(
            target, player=player, patch=patch, spell_card_id=card.card_id
        )
    player.last_tavern_spell_cast = card.card_id
    bump_seat_counter(player, SPELLS_CAST, patch=patch)
    bump_seat_counter(player, TAVERN_SPELLS_CAST)
    _fire_tavern_spell_cast(player, rng=rng, patch=patch, shared_pool=shared_pool)


def _resolve_spell_abilities(
    player: PlayerState,
    card: SpellCard,
    *,
    rng: np.random.Generator,
    patch: PatchContext,
    target: Optional[Minion],
    choose_one_option: int,
    shop_excluded_race: Optional[Race],
    shared_pool: Optional[SharedCardPool],
) -> None:
    for ability in card.abilities:
        if ability.trigger is Trigger.ON_START_OF_COMBAT:
            # "Start of Combat: …" on a spell. Nothing happens now; the seat
            # holds the promise and the next fight reads it. Registered per
            # cast, so a Bounty cast twice promises twice.
            player.start_combat_promises = player.start_combat_promises + (ability,)
            continue
        if ability.trigger != Trigger.ON_PLACE:
            continue
        _apply_spell_effect(
            player,
            ability.effect,
            rng=rng,
            patch=patch,
            target=target,
            choose_one_option=choose_one_option,
            shop_excluded_race=shop_excluded_race,
            shared_pool=shared_pool,
        )


def _fire_tavern_spell_cast(
    player: PlayerState,
    *,
    rng: np.random.Generator,
    patch: PatchContext,
    shared_pool: Optional[SharedCardPool],
) -> None:
    """"Whenever you cast a Tavern spell" — board listeners, left to right."""
    from .shop_triggers import ShopTriggers

    triggers = ShopTriggers(rng, patch=patch)
    for source in list(player.board):
        for ability in source.abilities:
            if ability.trigger != Trigger.ON_TAVERN_SPELL_CAST:
                continue
            triggers.apply_shop_effect(
                player, source, ability.effect, placed=None, shared_pool=shared_pool
            )


def apply_tavern_spell_effect(
    player: PlayerState,
    effect: object,
    *,
    rng: np.random.Generator,
    patch: PatchContext,
    source: Optional[Minion] = None,
    shop_excluded_race: Optional[Race] = None,
    shared_pool: Optional[SharedCardPool] = None,
) -> None:
    """Resolve a Tavern-spell effect carried by a *minion* rather than a spell.

    "Battlecry: Get a random Tavern spell", "Activate: Discover a Tavern spell",
    "Rally: your Tavern spells give an extra +1 Health" — the effect is the same
    one a spell would carry, so it goes through the same resolver. ``source`` is
    the minion, which is who a self-targeting cast aims at.
    """
    _apply_spell_effect(
        player,
        effect,
        rng=rng,
        patch=patch,
        target=source,
        choose_one_option=0,
        shop_excluded_race=shop_excluded_race,
        shared_pool=shared_pool,
    )


def _apply_spell_effect(
    player: PlayerState,
    effect: object,
    *,
    rng: np.random.Generator,
    patch: PatchContext,
    target: Optional[Minion],
    choose_one_option: int,
    shop_excluded_race: Optional[Race],
    shared_pool: Optional[SharedCardPool],
) -> None:
    from .shop_triggers import ShopTriggers
    from .targeted_battlecry import apply_targeted_buff

    if isinstance(effect, ChooseOneEffect):
        # The seat took one half; the other never happens.
        chosen = effect.first if int(choose_one_option) == 0 else effect.second
        _apply_spell_effect(
            player,
            chosen,
            rng=rng,
            patch=patch,
            target=target,
            choose_one_option=choose_one_option,
            shop_excluded_race=shop_excluded_race,
            shared_pool=shared_pool,
        )
        return

    if isinstance(effect, BuffTargetFriendlyBattlecry):
        # A spell has no body on the board, so there is no "self" to exclude
        # and no caster to read adjacency from — only the minion it names.
        # "Your Tavern spells give an extra +1 Attack" lands here, on the buff
        # itself, so it applies once however many minions the spell reaches.
        extra_attack, extra_health = tavern_spell_bonus(player)
        apply_targeted_buff(
            player,
            source=None,
            effect=replace(
                effect,
                attack=effect.attack + extra_attack,
                health=effect.health + extra_health,
            ),
            rng=rng,
            forced_buff_target=target,
        )
        return

    if isinstance(effect, RaiseStandingBonusEffect) and effect.scope_key_from_target:
        # "Choose a minion. Give minions of **its type** in the Tavern +3/+3
        # this game" — the scope is the target's tribe, so a spell with no
        # target, or one aimed at a tribeless body, names no scope at all.
        if target is None or target.race is None:
            return
        raise_standing_bonus(
            player,
            BonusScope(effect.scope_kind, target.race, effect.scope_max_tier),
            effect.attack,
            effect.health,
        )
        return

    if isinstance(effect, DiscoverHeroPowerEffect):
        from src.bg_lobby.player import PendingChoice, PendingChoiceKind

        # "a **new** Hero Power": the seat's own is the one exclusion.
        held = player.hero.hero_id if player.hero is not None else None
        pool = sorted(cid for cid in patch.hero_pool_ids if cid != held)
        if len(pool) < 3:
            return
        picks = []
        remaining = list(pool)
        for _ in range(3):
            picks.append(remaining.pop(int(rng.integers(0, len(remaining)))))
        player.pending_choice = PendingChoice(
            PendingChoiceKind.HERO_POWER_DISCOVER, tuple(picks), 0
        )
        return

    if isinstance(effect, SummonOnCombatSpaceEffect):
        # One entry per charge. They are the seat's and outlive a fight that
        # had no room to spend them.
        player.combat_space_summons = player.combat_space_summons + (
            (effect,) * max(1, int(effect.charges))
        )
        return

    if isinstance(effect, StealNeighbourBloodGemsEffect):
        from src.bg_recruitment.blood_gems import play_blood_gem_on

        if target is None or target not in player.board:
            return
        play_blood_gem_on(player, target, count=effect.gems, patch=patch)
        idx = player.board.index(target)
        for side in (idx - 1, idx + 1):
            if not 0 <= side < len(player.board):
                continue
            neighbour = player.board[side]
            attack, health = neighbour.blood_gem_attack, neighbour.blood_gem_health
            if not attack and not health:
                continue
            neighbour.bonus_attack -= attack
            neighbour.bonus_health -= health
            neighbour.blood_gem_attack = neighbour.blood_gem_health = 0
            target.bonus_attack += attack
            target.bonus_health += health
            target.blood_gem_attack += attack
            target.blood_gem_health += health
        return

    if isinstance(effect, RefreshWithTavernSpellsEffect):
        from src.bg_recruitment.shop import clear_shop_slot, effective_shop_offers_count

        # The minion row goes back to the lobby, and the counter shows as many
        # spells as it was showing cards.
        want = effective_shop_offers_count(player)
        for slot in range(len(player.shop)):
            clear_shop_slot(player, slot, shared_pool, release_to_pool=True)
        offer_tavern_spells(
            player,
            rng=rng,
            patch=patch,
            count=want,
            shop_excluded_race=shop_excluded_race,
        )
        return

    if isinstance(effect, RefreshWithTribeEffect):
        from src.bg_recruitment.shop import refresh_shop

        if target is None or target.race is None:
            return
        refresh_shop(
            player,
            shop_excluded_race,
            rng=rng,
            shared_pool=shared_pool,
            frozen_slots=player.shop_frozen,
            patch=patch,
            tribe=target.race,
        )
        return

    if isinstance(effect, BloodGemsOnEveryRefreshEffect):
        player.refresh_blood_gems += max(1, int(effect.count))
        return

    if isinstance(effect, DestroyFriendlyEffect):
        from .shop_triggers import ShopTriggers as _ShopTriggers
        from src.bg_recruitment.targeted_battlecry import apply_destroy_friendly

        # A spell has no body of its own, so there is no eater and no "self"
        # to exclude — only the friendly the seat named. This is the one effect
        # in ``_HANDLED_ELSEWHERE`` a Tavern spell carries, so without a branch
        # here Butchering was two gold for nothing at all.
        apply_destroy_friendly(
            player,
            None,
            effect,
            rng=rng,
            forced=target if target in player.board else None,
            triggers=_ShopTriggers(rng, patch=patch),
            shared_pool=shared_pool,
        )
        return

    if isinstance(effect, PromiseNextTurnEffect):
        # Nothing now. The seat remembers the promise and the body it named, if
        # it named one, and the start of its next turn is where it is asked.
        # Named by ``promise_tag`` and not by ``instance_id``: the seat's state
        # is copied once per action and the copy re-issues instance ids, so a
        # promise that noted one could never find its body again.
        tag = 0
        if target is not None:
            if not target.promise_tag:
                target.promise_tag = next_instance_id()
            tag = target.promise_tag
        player.next_turn_promises = player.next_turn_promises + ((effect, tag),)
        return

    if isinstance(effect, MakeFriendlyGoldenEffect):
        from src.bg_recruitment.targeted_battlecry import make_golden

        if effect.in_tavern:
            # "a **random** minion in the Tavern" — the seat names nobody.
            filled = [i for i, m in enumerate(player.shop) if m is not None]
            if not filled:
                return
            offer = player.shop[filled[int(rng.integers(0, len(filled)))]]
            # ``make_golden`` takes the two extra copies the offer will stand
            # for, and refuses if the lobby cannot cover them — including the
            # case where the pick is already Golden and there is nothing to do.
            make_golden(offer, patch=patch, shared_pool=shared_pool)
            return
        chosen = target if target is not None else _random_friendly(player, rng)
        if chosen is None:
            return
        if effect.max_tier and chosen.tier > effect.max_tier:
            return
        make_golden(chosen, patch=patch, shared_pool=shared_pool)
        return

    if isinstance(effect, TransformToHigherTierEffect):
        from src.bg_recruitment.faceless import transform_to_higher_tier

        chosen = target if target is not None else _random_friendly(player, rng)
        if chosen is None:
            return
        transform_to_higher_tier(
            chosen,
            rng=rng,
            patch=patch,
            tiers_up=effect.tiers_up,
            shop_excluded_race=shop_excluded_race,
            shared_pool=shared_pool,
        )
        return

    if isinstance(effect, SellFriendlyForStatsEffect):
        from .shop_triggers import ShopTriggers as _ShopTriggers
        from src.bg_recruitment.targeted_battlecry import apply_sell_friendly_for_stats

        victim = target if target in player.board else _random_friendly(player, rng)
        apply_sell_friendly_for_stats(
            player,
            victim,
            effect,
            rng=rng,
            triggers=_ShopTriggers(rng, patch=patch),
            shop_excluded_race=shop_excluded_race,
            shared_pool=shared_pool,
        )
        return

    if isinstance(effect, BuffAllShopOffersEffect):
        extra_attack, extra_health = tavern_spell_bonus(player)
        effect = replace(
            effect,
            attack=effect.attack + extra_attack,
            health=effect.health + extra_health,
        )

    if isinstance(effect, IncreaseTavernSpellBonusEffect):
        player.tavern_spell_bonus_attack += int(effect.attack)
        player.tavern_spell_bonus_health += int(effect.health)
        return

    if isinstance(effect, AddRandomTavernSpellToHandEffect):
        add_random_tavern_spells(
            player,
            count=effect.count,
            max_cost=effect.max_cost,
            gives_stats=effect.gives_stats,
            spellcraft=effect.spellcraft,
            rng=rng,
            patch=patch,
            shop_excluded_race=shop_excluded_race,
        )
        return

    if isinstance(effect, CopyLastTavernSpellEffect):
        last = player.last_tavern_spell_cast
        if last is not None:
            for _ in range(max(1, effect.count)):
                _give_spell(player, patch.tavern_spells.get(last))
        return

    if isinstance(effect, CastRandomTavernSpellEffect):
        pool = tavern_spell_pool(
            player.tavern_tier, patch=patch, shop_excluded_race=shop_excluded_race
        )
        if pool:
            rolled = patch.tavern_spells[pool[int(rng.integers(0, len(pool)))]]
            cast_tavern_spell(
                player,
                rolled,
                rng=rng,
                patch=patch,
                target=target,
                shop_excluded_race=shop_excluded_race,
                shared_pool=shared_pool,
            )
        return

    if isinstance(effect, StealTavernMinionEffect):
        steal_tavern_minion(
            player,
            rng=rng,
            shared_pool=shared_pool,
            highest_attack=effect.highest_attack,
        )
        return

    if isinstance(effect, BuffSharedTribeEffect):
        if target is not None:
            for friendly in player.board:
                if minion_matches_tribe(friendly, target.race):
                    friendly.bonus_attack += effect.attack
                    friendly.bonus_health += effect.health
        return

    if isinstance(effect, CastSpellAtEffect):
        # ``target`` is the minion carrying this effect — the positions the card
        # names are read from where *it* stands.
        spell = patch.tavern_spells.get(effect.card_id)
        if spell is not None and target is not None:
            aimed_at = (
                [None] if effect.untargeted else _positional_targets(player, target, effect)
            )
            for _ in range(max(1, effect.repeats)):
                for aimed in aimed_at:
                    cast_tavern_spell(
                        player,
                        spell,
                        rng=rng,
                        patch=patch,
                        target=aimed,
                        shop_excluded_race=shop_excluded_race,
                        shared_pool=shared_pool,
                    )
        return

    if isinstance(effect, DiscoverTavernSpellEffect):
        open_tavern_spell_discover(
            player,
            rng=rng,
            patch=patch,
            repeats=effect.repeats,
            exact_tier=effect.exact_tier,
            shop_excluded_race=shop_excluded_race,
        )
        return

    if isinstance(effect, DiscoverMinionAtTierEffect):
        _open_tier_discover(
            player,
            effect.tier * improve_level(player, effect.counter, effect.per),
            rng=rng,
            patch=patch,
            shop_excluded_race=shop_excluded_race,
            shared_pool=shared_pool,
        )
        return

    ShopTriggers(rng, patch=patch).apply_shop_effect(
        player,
        # The minion the spell was cast at, where it had one: a spell has no
        # body of its own, so "this minion" in an effect means its target.
        source=target,
        effect=effect,
        placed=None,
        shop_excluded_race=shop_excluded_race,
        shared_pool=shared_pool,
    )


def _random_friendly(player: PlayerState, rng: np.random.Generator) -> Optional[Minion]:
    """A friendly at random — what a targeted spell falls back to with no pick.

    Same arrangement every other targeted effect has: the seat's choice when it
    made one, a legal target otherwise.
    """
    if not player.board:
        return None
    return player.board[int(rng.integers(0, len(player.board)))]


def _positional_targets(player: PlayerState, source: Minion, effect) -> List[Minion]:
    """Who a positional cast lands on — the card names them, not the seat."""
    board = player.board
    try:
        idx = board.index(source)
    except ValueError:
        return []
    if effect.to_the_right:
        return [board[idx + 1]] if idx + 1 < len(board) else []
    if effect.adjacent:
        return [board[i] for i in (idx - 1, idx + 1) if 0 <= i < len(board)]
    return [source]


def steal_tavern_minion(
    player: PlayerState,
    *,
    rng: np.random.Generator,
    shared_pool: Optional[SharedCardPool] = None,
    highest_attack: bool = False,
) -> Optional[Minion]:
    """Take one minion off the counter into hand, free.

    Enchanted Lasso takes a random one, Decoy Conjurer the biggest; that is the
    only difference between them, so it is a flag rather than two functions.
    """
    filled = [i for i, m in enumerate(player.shop) if m is not None]
    slot = first_free_hand_slot(player)
    if not filled or slot is None:
        return None
    if highest_attack:
        idx = max(filled, key=lambda i: player.shop[i].raw_attack)
    else:
        idx = filled[int(rng.integers(0, len(filled)))]
    taken = player.shop[idx]
    player.shop[idx] = None
    player.hand[slot] = taken
    # It left the tavern for a hand, which is the same thing a purchase does to
    # the shared pool even though no gold moved.
    on_bought_from_shop(shared_pool, taken)
    return taken


def _give_spell(player: PlayerState, spell: Optional[SpellCard]) -> bool:
    """Put ``spell`` in the first free hand slot. False if it does not fit."""
    if spell is None:
        return False
    slot = first_free_hand_slot(player)
    if slot is None:
        return False
    player.hand[slot] = spell
    return True


def spellcraft_spell_ids(patch: PatchContext) -> set:
    """Every Spellcraft spell the package's Nagas can make.

    Gathered from the bindings rather than the catalog: a Spellcraft spell is
    described by the minion that mints it, so what exists is what some card
    says it makes.
    """
    from src.bg_core.effects import CreateSpellcraftSpellEffect

    out = set()
    for abilities in patch.effects.values():
        for ability in abilities:
            effect = ability.effect
            if isinstance(effect, CreateSpellcraftSpellEffect) and effect.card_id:
                out.add(effect.card_id)
    return out


def add_random_tavern_spells(
    player: PlayerState,
    *,
    count: int = 1,
    max_cost: int = 0,
    gives_stats: bool = False,
    spellcraft: bool = False,
    rng: np.random.Generator,
    patch: PatchContext,
    shop_excluded_race=None,
) -> int:
    """"Get N random Tavern spells", filtered as the card prints it.

    Returns how many actually fit in hand — a full hand takes what it can, the
    same as every other card-giving effect.

    ``spellcraft`` draws from a different pool entirely: a Spellcraft spell is
    minted by a Naga rather than offered on the counter, so it is never in the
    tavern's own list and has to be gathered from the cards that make one.
    """
    ctx = require_patch(patch, where="tavern_spells.add_random_tavern_spells")
    if spellcraft:
        pool = sorted(spellcraft_spell_ids(ctx))
    else:
        pool = [
            card_id
            for card_id in tavern_spell_pool(
                player.tavern_tier, patch=ctx, shop_excluded_race=shop_excluded_race
            )
            if (not max_cost or ctx.tavern_spells[card_id].cost <= max_cost)
            and (not gives_stats or spell_gives_stats(ctx.tavern_spells[card_id]))
        ]
    if not pool:
        return 0
    given = 0
    for _ in range(max(0, int(count))):
        pick = pool[int(rng.integers(0, len(pool)))]
        if not _give_spell(player, ctx.tavern_spells[pick]):
            break
        given += 1
    return given


def open_tavern_spell_discover(
    player: PlayerState,
    *,
    rng: np.random.Generator,
    patch: PatchContext,
    repeats: int = 1,
    exact_tier: bool = False,
    shop_excluded_race=None,
) -> bool:
    """"Discover a Tavern spell": three offered, the seat keeps one.

    No shared-pool reservation, unlike a minion Discover — Tavern spells are
    not drawn from the lobby's shared minion pool, so there is nothing to
    reserve or hand back.
    """
    from src.bg_lobby.player import PendingChoiceKind

    from .discover import try_open_hand_discover_modal

    ctx = require_patch(patch, where="tavern_spells.open_tavern_spell_discover")
    pool = tavern_spell_pool(
        player.tavern_tier, patch=ctx, shop_excluded_race=shop_excluded_race
    )
    if exact_tier:
        # "of your Tier" — the seat's own, not everything up to it.
        at_tier = [c for c in pool if ctx.tavern_spells[c].tier == player.tavern_tier]
        if len(at_tier) >= 3:
            pool = at_tier
    if len(pool) < 3:
        return False
    picks: List[str] = []
    remaining = list(pool)
    for _ in range(3):
        picks.append(remaining.pop(int(rng.integers(0, len(remaining)))))
    return try_open_hand_discover_modal(
        player, PendingChoiceKind.SPELL_DISCOVER, tuple(picks), max(1, repeats) - 1
    )


def _open_tier_discover(
    player: PlayerState,
    tier: int,
    *,
    rng: np.random.Generator,
    patch: PatchContext,
    shop_excluded_race: Optional[Race],
    shared_pool: Optional[SharedCardPool],
) -> None:
    """A New Sprout: three minions of one tier, the seat keeps one.

    The tier is the card's own, so it is not held to the tavern's ceiling:
    Hallowed Ritual says Tier 7 in a game whose tavern stops at 6.
    """
    from src.bg_lobby.player import PendingChoiceKind

    from .discover import try_open_hand_discover_modal
    from .discover_pool import roll_triple_reward_discover_at_target_tier

    options = roll_triple_reward_discover_at_target_tier(
        rng,
        tier,
        shop_excluded_race,
        shared_pool=shared_pool,
        patch=patch,
        printed_tier=True,
    )
    if options is None:
        return
    try_open_hand_discover_modal(
        player,
        PendingChoiceKind.TAVERN_SPELL_DISCOVER,
        options,
        0,  # one Discover, no chain behind it
        shared_pool=shared_pool,
    )
