"""Shop economy: buy, sell, roll, level up."""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from src.bg_catalog.patch_context import PatchContext
from src.bg_catalog.ruleset import Ruleset
from src.bg_core.minion import Minion, Race

from src.envs.minibg.actions import (
    BUY_COST,
    MAX_SHOP_SLOTS,
    ROLL_COST,
    SELL_REWARD,
    shop_offers_count,
)
from src.bg_lobby.player import PlayerState
from src.bg_lobby.shared_pool import SharedCardPool

from src.bg_core.conditions import condition_met
from src.bg_core.effects import SellValueEffect, Trigger

from .hand_slots import first_free_hand_slot
from .pool_ledger import on_sell_minion
from .shop import clear_shop_slot, fill_shop_slot, refresh_shop, tavern_card_pool


def _ruleset(player: Optional[PlayerState]):
    """The seat's own numbers, or the defaults when there is no seat to ask.

    ``buy_cost``/``sell_reward``/``roll_cost`` were fields on Ruleset that
    nothing read: the module constants were used directly, so a package
    setting them was silently ignored. They happen to agree today, which is
    exactly why it went unnoticed.
    """
    from src.bg_catalog.ruleset import DEFAULT_RULESET

    rs = getattr(player, "ruleset", None) if player is not None else None
    return rs if rs is not None else DEFAULT_RULESET


def effective_sell_reward(minion: Minion, player: Optional[PlayerState] = None) -> int:
    """What this minion is worth on the way out.

    Three prices, most specific first: one the card prints behind a condition
    (Tortollan Blue Shell, which needs the seat to answer it), one the card
    prints flat, and the ruleset's.
    """
    for ab in minion.abilities:
        if ab.trigger is not Trigger.AURA or not isinstance(ab.effect, SellValueEffect):
            continue
        if ab.condition is not None:
            if player is None or not condition_met(ab.condition, player, player.board):
                continue
        return int(ab.effect.amount)
    if minion.sell_value is not None:
        return int(minion.sell_value)
    return _ruleset(player).sell_reward


def effective_roll_cost(player: PlayerState) -> int:
    # Nozdormu: first refresh each turn is free (takes precedence).
    if player.hero_free_roll_pending:
        return 0
    if player.next_roll_cost_override is not None:
        return max(0, int(player.next_roll_cost_override))
    # Millhouse: every refresh costs a flat amount.
    if player.hero is not None:
        flat = player.hero.flat_refresh_cost()
        if flat is not None:
            return max(0, flat)
    return _ruleset(player).roll_cost


def effective_buy_cost(player: PlayerState) -> int:
    # Aranna: the first buy each turn is free once her power unlocks. Answered
    # here rather than at the purchase so the legal mask agrees — a seat with
    # no gold and a free buy can still buy.
    if player.hero_free_buys > 0:
        return 0
    # Millhouse: minions cost a flat amount.
    if player.hero is not None:
        flat = player.hero.flat_buy_cost()
        if flat is not None:
            return max(0, flat)
    return _ruleset(player).buy_cost


def effective_level_up_cost(player: PlayerState) -> int:
    base = player.next_tier_up_cost + player.upgrade_cost_delta
    if player.hero is not None:
        base += player.hero.upgrade_cost_surcharge()  # Millhouse: +1
        base -= player.hero_upgrade_discount  # Chenvaala: accumulated discount
    return max(0, base)


def _pay_refresh_in_health(player: PlayerState, cost: int, *, patch=None) -> bool:
    """Pay this refresh in Health if the seat has a charge for it.

    The payment is hero damage rather than a bare subtraction, so everything
    that reads hero damage sees it: armor absorbs it first, and a card that
    undoes hero damage undoes this one too — which is the same rule the card
    would face from a combat.
    """
    from src.bg_core.effects import RefreshesCostHealthEffect, Trigger
    from src.bg_lobby.player import apply_hero_damage

    if player.health_refreshes_left <= 0 or cost <= 0:
        return False
    amount = next(
        (
            ability.effect.amount
            for minion in player.board
            for ability in minion.abilities
            if ability.trigger is Trigger.AURA
            and isinstance(ability.effect, RefreshesCostHealthEffect)
        ),
        None,
    )
    if amount is None:
        player.health_refreshes_left = 0
        return False
    player.health_refreshes_left -= 1
    apply_hero_damage(player, int(amount), patch=patch)
    return True


def reset_health_refreshes(player: PlayerState) -> None:
    """Give back this turn's health-paid refreshes, from what is on the board."""
    from src.bg_core.effects import RefreshesCostHealthEffect, Trigger

    player.health_refreshes_left = max(
        (
            ability.effect.uses
            for minion in player.board
            for ability in minion.abilities
            if ability.trigger is Trigger.AURA
            and isinstance(ability.effect, RefreshesCostHealthEffect)
        ),
        default=0,
    )


def start_of_turn_gold(player: PlayerState, round_number: int) -> int:
    """The coins a seat starts ``round_number`` with, banked promises included.

    Reads and clears ``gold_next_turn``: "Gain 1 Gold next turn" is paid exactly
    once, by the turn it named. Both lobby types call this instead of setting
    ``gold`` from the curve directly, so a promise cannot be honoured in one and
    dropped in the other.
    """
    banked = max(0, int(player.gold_next_turn))
    player.gold_next_turn = 0
    return player.ruleset.gold_for_round(int(round_number)) + banked


def accrue_upgrade_discount(player: PlayerState) -> None:
    """Waiting a round makes the next tier cheaper (the standing BG discount).

    Called once per player at the start of each round. Lived as two identical
    copies — one per lobby type — which is how the price came to have several
    writers in the first place.

    Reads the seat's own ruleset rather than taking one: accruing at one
    package's rate while pricing from another's table is precisely the split
    this whole change exists to remove.
    """
    ruleset = player.ruleset
    if player.tavern_tier >= ruleset.max_tier:
        return  # no next tier to discount
    player.upgrade_discount_accrued += ruleset.level_up_discount_per_round


def sell_from_board(
    player: PlayerState,
    pos: int,
    *,
    on_sell: Callable[[Minion, PlayerState], None] | None = None,
    on_triples: Callable[[PlayerState], None],
    shared_pool: Optional[SharedCardPool] = None,
) -> None:
    sold = player.board[pos]
    if on_sell is not None:
        on_sell(sold, player)
    on_sell_minion(shared_pool, sold)
    del player.board[pos]
    player.gold += effective_sell_reward(sold, player)
    on_triples(player)


def note_gold_spent(player: PlayerState, amount: int, *, patch=None) -> None:
    """Tell the board that gold left the seat.

    One helper at every spend site rather than a hook per action, because the
    cards say "after you spend N Gold" without caring what it went on — and a
    Tavern spell and an Activate are both gold going out. Those two paid out of
    ``player.gold`` directly and so were invisible to every watcher.
    """
    if amount <= 0 or patch is None:
        return
    from src.bg_core.board_helpers import seat_rng

    from .hero_passives import apply_hero_on_gold_spent
    from .shop_triggers import ShopTriggers

    apply_hero_on_gold_spent(player, amount)
    ShopTriggers(seat_rng(player), patch=patch).fire_gold_spent(player, amount)


def buy_from_shop(
    player: PlayerState,
    slot: int,
    *,
    patch=None,
    on_bought: Callable[[Minion, PlayerState], None],
    on_friendly_bought: Callable[[Minion, PlayerState], None] | None = None,
    on_triples: Callable[[PlayerState], None],
    shared_pool: Optional[SharedCardPool] = None,
) -> None:
    minion = player.shop[slot]
    assert minion is not None
    buy_cost = effective_buy_cost(player)
    # Spent by the purchase, not by quoting the price: re-granted at the next
    # turn start for as long as the power stays unlocked.
    if player.hero_free_buys > 0:
        player.hero_free_buys -= 1
    player.gold -= buy_cost
    clear_shop_slot(player, slot, shared_pool, release_to_pool=False)
    h = first_free_hand_slot(player)
    assert h is not None, "BUY illegal when hand is full (legal mask bug)"
    player.hand[h] = minion
    # Only once the minion is safely in hand. "After you spend N Gold" can hand
    # the seat a card, and firing it first let a watcher take the last slot out
    # from under the purchase the mask had already ruled legal -- gold paid,
    # shop slot cleared, minion gone, and an assertion where a card should be.
    note_gold_spent(player, buy_cost, patch=patch)
    on_bought(minion, player)
    if on_friendly_bought is not None:
        on_friendly_bought(minion, player)
    on_triples(player)


def roll_shop(
    player: PlayerState,
    shop_excluded_race: Optional[Race],
    *,
    rng: np.random.Generator,
    shared_pool: Optional[SharedCardPool] = None,
    patch: PatchContext,
) -> None:
    cost = effective_roll_cost(player)
    paid_in_health = _pay_refresh_in_health(player, cost, patch=patch)
    if not paid_in_health:
        player.gold -= cost
        note_gold_spent(player, cost, patch=patch)
    # Nozdormu: consume the free first refresh for this turn.
    if player.hero_free_roll_pending:
        player.hero_free_roll_pending = False
    if player.free_roll_charges > 0:
        player.free_roll_charges -= 1
        if player.free_roll_charges > 0:
            player.next_roll_cost_override = 0
        else:
            player.next_roll_cost_override = None
    elif player.next_roll_cost_override is not None:
        player.next_roll_cost_override = None
    refresh_shop(
        player,
        shop_excluded_race,
        rng=rng,
        shared_pool=shared_pool,
        frozen_slots=player.shop_frozen,
        patch=patch,
    )


def level_up_tavern(
    player: PlayerState,
    shop_excluded_race: Optional[Race],
    *,
    rng: np.random.Generator,
    shared_pool: Optional[SharedCardPool] = None,
    patch: PatchContext,
) -> None:
    cost = effective_level_up_cost(player)
    player.gold -= cost
    note_gold_spent(player, cost, patch=patch)
    player.upgrade_cost_delta = 0
    player.hero_upgrade_discount = 0  # Chenvaala: discount consumed by the upgrade
    old_tier = player.tavern_tier
    player.tavern_tier += 1
    # The price of the tier just reached is the package's, and the rounds spent
    # waiting for the previous one do not pre-discount it.
    player.upgrade_discount_accrued = 0
    extra = player.hero.extra_shop_slots() if player.hero is not None else 0
    old_n = min(MAX_SHOP_SLOTS, shop_offers_count(old_tier) + extra)
    new_n = min(MAX_SHOP_SLOTS, shop_offers_count(player.tavern_tier) + extra)
    while len(player.shop) < MAX_SHOP_SLOTS:
        player.shop.append(None)
    for i in range(old_n, new_n):
        fill_shop_slot(
            player,
            i,
            shop_excluded_race,
            rng=rng,
            shared_pool=shared_pool,
            patch=patch,
        )
