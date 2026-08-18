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

from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.bg_catalog.patch_context import PatchContext, require_patch
from src.bg_core.effects import (
    BuffTargetFriendlyBattlecry,
    ChooseOneEffect,
    DiscoverMinionAtTierEffect,
    StealTavernMinionEffect,
    Trigger,
)
from src.bg_core.minion import Minion, Race
from src.bg_core.spell_card import SpellCard
from src.bg_lobby.player import PlayerPhase, PlayerState
from src.bg_lobby.shared_pool import SharedCardPool

from .hand_slots import first_free_hand_slot
from .pool_ledger import on_bought_from_shop

__all__ = [
    "TavernSpellNotAllowed",
    "tavern_spell_pool",
    "effective_tavern_spell_cost",
    "offer_tavern_spells",
    "clear_tavern_spell_offers",
    "buy_tavern_spell",
    "steal_tavern_minion",
    "play_tavern_spell_from_hand",
]


class TavernSpellNotAllowed(ValueError):
    """The seat cannot do this with a Tavern spell right now, and why."""


def tavern_spell_pool(tavern_tier: int, *, patch: PatchContext) -> List[str]:
    """Spell ids the tavern can offer a seat at ``tavern_tier``.

    Same rule as the minion counter: everything up to the seat's tier, so a
    tier-1 spell keeps showing up all game.
    """
    ctx = require_patch(patch, where="tavern_spells.tavern_spell_pool")
    return sorted(
        card_id
        for card_id, spell in ctx.tavern_spells.items()
        if spell.is_tavern_spell and 1 <= spell.tier <= int(tavern_tier)
    )


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
        want = max(0, int(player.ruleset.tavern_spells_per_roll))
        pool = tavern_spell_pool(player.tavern_tier, patch=ctx)
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
    if player.gold < cost:
        raise TavernSpellNotAllowed(
            f"{spell.card_id} costs {cost}; the seat has {player.gold}"
        )
    slot = first_free_hand_slot(player)
    if slot is None:
        raise TavernSpellNotAllowed("hand is full")

    player.gold -= cost
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
    choose_one_option: int = 0,
    shop_excluded_race: Optional[Race] = None,
    shared_pool: Optional[SharedCardPool] = None,
) -> None:
    """Cast the Tavern spell in ``hand_index``, then discard it.

    ``target_board_index`` is the friendly a "give a minion +X/+Y" names, and
    ``choose_one_option`` picks the half of a Choose One. Both are the seat's
    decisions; with neither given the effect falls back to a random legal
    target, which is what the placement path already does.
    """
    ctx = require_patch(patch, where="tavern_spells.play_tavern_spell_from_hand")
    card = player.hand[hand_index] if 0 <= hand_index < len(player.hand) else None
    if not isinstance(card, SpellCard) or not card.is_tavern_spell:
        raise TavernSpellNotAllowed(f"hand slot {hand_index} holds no Tavern spell")
    if player.phase != PlayerPhase.SHOP:
        raise TavernSpellNotAllowed("casting is a recruit-phase move")

    target = (
        player.board[target_board_index]
        if target_board_index is not None and 0 <= target_board_index < len(player.board)
        else None
    )
    player.hand[hand_index] = None
    for ability in card.abilities:
        if ability.trigger != Trigger.ON_PLACE:
            continue
        _apply_spell_effect(
            player,
            ability.effect,
            rng=rng,
            patch=ctx,
            target=target,
            choose_one_option=choose_one_option,
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
        apply_targeted_buff(
            player,
            source=None,
            effect=effect,
            rng=rng,
            forced_buff_target=target,
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

    if isinstance(effect, DiscoverMinionAtTierEffect):
        _open_tier_discover(
            player,
            effect.tier,
            rng=rng,
            patch=patch,
            shop_excluded_race=shop_excluded_race,
            shared_pool=shared_pool,
        )
        return

    ShopTriggers(rng, patch=patch).apply_shop_effect(
        player,
        source=None,
        effect=effect,
        placed=None,
        shop_excluded_race=shop_excluded_race,
        shared_pool=shared_pool,
    )


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


def _open_tier_discover(
    player: PlayerState,
    tier: int,
    *,
    rng: np.random.Generator,
    patch: PatchContext,
    shop_excluded_race: Optional[Race],
    shared_pool: Optional[SharedCardPool],
) -> None:
    """A New Sprout: three minions of one tier, the seat keeps one."""
    from src.bg_lobby.player import PendingChoiceKind

    from .discover import try_open_hand_discover_modal
    from .discover_pool import roll_triple_reward_discover_at_target_tier

    options = roll_triple_reward_discover_at_target_tier(
        rng,
        tier,
        shop_excluded_race,
        shared_pool=shared_pool,
        patch=patch,
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
