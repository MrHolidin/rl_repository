"""Discover / Adapt pending-choice resolution."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from src.bg_core.effects import DiscoverMurlocEffect, Trigger
from src.bg_core.minion import Minion, Race
from src.envs.minibg.actions import HAND_SIZE
from src.envs.minibg.discover_pool import (
    apply_adapt_key_to_minion,
    is_murloc_board_minion,
    roll_adapt_triple,
    roll_discover_murloc_triple,
    roll_triple_reward_discover_triple,
)
from src.envs.minibg.state import PendingChoice, PendingChoiceKind, PlayerState

from .triples import flush_triple_reward_queue_if_idle, hand_has_free_slot, resolve_triples_loop


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
) -> PendingChoice:
    if kind == PendingChoiceKind.DISCOVER_MURLOC:
        opts = roll_discover_murloc_triple(
            rng, player.tavern_tier, shop_excluded_race
        )
    elif kind == PendingChoiceKind.TRIPLE_REWARD_DISCOVER:
        opts = roll_triple_reward_discover_triple(
            rng, player.tavern_tier, shop_excluded_race
        )
    else:
        opts = roll_adapt_triple(rng)
    return PendingChoice(kind, opts, remaining_after)


def resolve_discover_pick(
    player: PlayerState,
    pick_slot: int,
    shop_excluded_race: Optional[Race],
    *,
    rng: np.random.Generator,
    on_after_placed: Callable[[PlayerState, Minion], None],
) -> None:
    from src.bg_catalog.cards import make_minion

    pc = player.pending_choice
    assert pc is not None
    assert 0 <= pick_slot <= 2
    choice_token = pc.options[pick_slot]
    extra = pc.extra_modals_after
    hand_discover = pc.kind in (
        PendingChoiceKind.DISCOVER_MURLOC,
        PendingChoiceKind.TRIPLE_REWARD_DISCOVER,
    )
    if hand_discover:
        h = next((i for i in range(HAND_SIZE) if player.hand[i] is None), None)
        if h is None:
            raise ValueError(
                "DISCOVER pick with full hand; legal mask must require a free hand slot"
            )
        player.hand[h] = make_minion(choice_token)
        resolve_triples_loop(player)
    else:
        for m in player.board:
            if is_murloc_board_minion(m):
                apply_adapt_key_to_minion(m, choice_token)

    chain_next = extra > 0
    if chain_next and hand_discover and not hand_has_free_slot(player):
        chain_next = False

    if chain_next:
        player.pending_choice = roll_pending_modal(
            rng, player, pc.kind, extra - 1, shop_excluded_race
        )
    else:
        player.pending_choice = None
        ref = player.placed_minion_pending_after
        if ref is not None:
            if ref in player.board:
                on_after_placed(player, ref)
            player.placed_minion_pending_after = None
            player.placed_minion_board_index = None
        flush_triple_reward_queue_if_idle(player, shop_excluded_race, rng=rng)
