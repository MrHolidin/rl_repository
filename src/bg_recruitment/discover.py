"""Discover / Adapt pending-choice resolution."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from src.bg_catalog.patch_context import PatchContext

from src.bg_core.effects import DiscoverMurlocEffect, Trigger
from src.bg_core.minion import Minion, Race
from .hand_slots import first_free_hand_slot
from src.bg_recruitment.discover_pool import (
    apply_adapt_key_to_minion,
    is_murloc_board_minion,
    roll_adapt_triple,
    roll_discover_murloc_triple,
    roll_triple_reward_discover_triple,
)
from src.bg_lobby.player import PendingChoice, PendingChoiceKind, PlayerState

from src.bg_lobby.shared_pool import SharedCardPool

from .pool_ledger import (
    on_discover_to_hand,
    release_discover_options,
    reserve_discover_options,
)
from .triples import flush_triple_reward_queue_if_idle, hand_has_free_slot, resolve_triples_loop

HAND_DISCOVER_KINDS = frozenset(
    {
        PendingChoiceKind.DISCOVER_MURLOC,
        PendingChoiceKind.TRIPLE_REWARD_DISCOVER,
    }
)


def is_hand_discover_kind(kind: PendingChoiceKind) -> bool:
    return kind in HAND_DISCOVER_KINDS


def is_hand_discover_pending(player: PlayerState) -> bool:
    pc = player.pending_choice
    return pc is not None and is_hand_discover_kind(pc.kind)


def try_open_hand_discover_modal(
    player: PlayerState,
    kind: PendingChoiceKind,
    options: tuple,
    extra_modals_after: int,
    *,
    shared_pool: Optional[SharedCardPool] = None,
) -> bool:
    """Open a hand-discover modal only if the player can take at least one pick now."""
    if not is_hand_discover_kind(kind):
        raise ValueError(f"not a hand-discover kind: {kind!r}")
    if not hand_has_free_slot(player):
        return False
    reserved = False
    if shared_pool is not None:
        if not reserve_discover_options(shared_pool, options):
            return False
        reserved = True
    player.pending_choice = PendingChoice(
        kind,
        options,
        extra_modals_after,
        options_pool_reserved=reserved,
    )
    return True


def try_open_hand_discover_chain(
    player: PlayerState,
    kind: PendingChoiceKind,
    extra_modals_after: int,
    shop_excluded_race: Optional[Race],
    *,
    rng: np.random.Generator,
    shared_pool: Optional[SharedCardPool] = None,
    patch: PatchContext,
) -> bool:
    """Roll and open the next hand-discover modal; reject if no hand slot."""
    if not is_hand_discover_kind(kind):
        raise ValueError(f"not a hand-discover kind: {kind!r}")
    if not hand_has_free_slot(player):
        return False
    pc = roll_pending_modal(
        rng,
        player,
        kind,
        extra_modals_after,
        shop_excluded_race,
        shared_pool=shared_pool,
        patch=patch,
    )
    if pc is None:
        return False
    player.pending_choice = pc
    return True


def discover_cards_to_receive(placed: Minion, battlecry_mult: int) -> int:
    n_need = 0
    for ab in placed.abilities:
        if ab.trigger != Trigger.ON_PLACE:
            continue
        if isinstance(ab.effect, DiscoverMurlocEffect):
            n_need = max(n_need, battlecry_mult * ab.effect.repeats)
    return n_need


def roll_pending_modal(
    rng: np.random.Generator,
    player: PlayerState,
    kind: PendingChoiceKind,
    remaining_after: int,
    shop_excluded_race: Optional[Race],
    *,
    shared_pool: Optional[SharedCardPool] = None,
    patch: PatchContext,
) -> Optional[PendingChoice]:
    if kind == PendingChoiceKind.DISCOVER_MURLOC:
        opts = roll_discover_murloc_triple(
            rng,
            player.tavern_tier,
            shop_excluded_race,
            shared_pool=shared_pool,
            patch=patch,
        )
    elif kind == PendingChoiceKind.TRIPLE_REWARD_DISCOVER:
        opts = roll_triple_reward_discover_triple(
            rng,
            player.tavern_tier,
            shop_excluded_race,
            shared_pool=shared_pool,
            patch=patch,
        )
    else:
        opts = roll_adapt_triple(rng)
    if opts is None:
        return None
    reserved = False
    if is_hand_discover_kind(kind) and shared_pool is not None:
        if not reserve_discover_options(shared_pool, opts):
            return None
        reserved = True
    return PendingChoice(kind, opts, remaining_after, options_pool_reserved=reserved)


def resolve_discover_pick(
    player: PlayerState,
    pick_slot: int,
    shop_excluded_race: Optional[Race],
    *,
    rng: np.random.Generator,
    on_after_placed: Callable[[PlayerState, Minion], None],
    shared_pool: Optional[SharedCardPool] = None,
    patch: PatchContext,
) -> None:
    from src.bg_catalog.cards import make_minion

    pc = player.pending_choice
    assert pc is not None
    assert 0 <= pick_slot <= 2
    choice_token = pc.options[pick_slot]
    extra = pc.extra_modals_after
    hand_discover = is_hand_discover_kind(pc.kind)
    if hand_discover:
        h = first_free_hand_slot(player)
        if h is None:
            raise ValueError(
                "DISCOVER pick with full hand; legal mask must require a free hand slot"
            )
        picked = make_minion(choice_token, patch=patch)
        player.hand[h] = picked
        if pc.options_pool_reserved and shared_pool is not None:
            release_discover_options(shared_pool, pc.options, keep_slot=pick_slot)
        else:
            on_discover_to_hand(shared_pool, picked)
    else:
        for m in player.board:
            if is_murloc_board_minion(m):
                apply_adapt_key_to_minion(m, choice_token)

    chain_next = extra > 0
    if chain_next:
        if hand_discover:
            if not try_open_hand_discover_chain(
                player,
                pc.kind,
                extra - 1,
                shop_excluded_race,
                rng=rng,
                shared_pool=shared_pool,
                patch=patch,
            ):
                chain_next = False
        else:
            player.pending_choice = roll_pending_modal(
                rng,
                player,
                pc.kind,
                extra - 1,
                shop_excluded_race,
                shared_pool=shared_pool,
                patch=patch,
            )
            if player.pending_choice is None:
                chain_next = False

    if not chain_next:
        player.pending_choice = None
        if hand_discover:
            resolve_triples_loop(player, shared_pool=shared_pool, patch=patch)
        ref = player.placed_minion_pending_after
        if ref is not None:
            if ref in player.board:
                on_after_placed(player, ref)
            player.placed_minion_pending_after = None
            player.placed_minion_board_index = None
        flush_triple_reward_queue_if_idle(
            player, shop_excluded_race, rng=rng, patch=patch
        )
