"""Transforming a body into a different card, keeping what the card says to.

Two shapes, and the difference is what survives. Faceless Taverngoer becomes a
shop offer outright — a fresh printing, nothing carried. Robust Evolution
becomes a random card from a Tier higher and *keeps its stats*, which is the
whole reason anyone casts it.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from src.bg_catalog.cards import make_minion
from src.bg_catalog.patch_context import PatchContext
from src.bg_core.effects import Keyword
from src.bg_core.minion import Minion, Race
from src.bg_lobby.player import PendingChoice, PendingChoiceKind, PlayerState


def filled_shop_slot_indices(player: PlayerState) -> Tuple[int, ...]:
    return tuple(i for i, m in enumerate(player.shop) if m is not None)


def transform_to_higher_tier(
    minion: Minion,
    *,
    rng: np.random.Generator,
    patch: PatchContext,
    tiers_up: int = 1,
    shop_excluded_race: Optional[Race] = None,
    shared_pool=None,
) -> bool:
    """Turn ``minion`` into a random card from ``tiers_up`` Tiers higher.

    In place, the way ``make_golden`` is: the body keeps its identity as far as
    the rest of the game is concerned (a spell aimed at it still hits, a hand
    slot still holds it), it is simply a different card now.

    The stats it had are folded into the new card's printing rather than left
    as bonuses. What granted them — a game-long tally, Blood Gems, a magnet —
    belonged to the card that is gone, and leaving those records behind would
    let the tally that fed the old card claw its own contribution back out of
    the new one. The seat's "this game" table is the exception: what the body
    already absorbed it keeps absorbed, so a scope it still matches does not
    pay twice and one it now matches for the first time pays once.

    Returns False when there is no card at that Tier to become — a Tier-7 body
    has nowhere higher to go.
    """
    from src.bg_recruitment.discover_pool import draw_from_pool, shop_pool_for_tier

    target_tier = int(minion.tier) + max(1, int(tiers_up))
    eligible = shop_pool_for_tier(
        target_tier, shop_excluded_race=shop_excluded_race, patch=patch
    )
    if shared_pool is not None:
        eligible = [cid for cid in eligible if shared_pool.remaining_copies(cid) > 0]
    if not eligible:
        return False
    picked = draw_from_pool(rng, eligible, 1, shared_pool=shared_pool)[0]
    # The seat stops holding one card and starts holding another, so the lobby
    # has to hear about both. Taken before given, so a lobby that cannot lend
    # the new card leaves the body — and the ledger — exactly as they were. A
    # Golden body was worth three copies of what it was and comes back plain,
    # which is three released for one taken.
    if shared_pool is not None:
        from src.bg_lobby.shared_pool import copies_for_minion

        if not shared_pool.acquire_new(picked, 1):
            return False
        shared_pool.release_offer(minion.card_id, copies_for_minion(minion))
    fresh = make_minion(picked, patch=patch)
    attack, health = minion.raw_attack, minion.max_health
    minion.card_id = fresh.card_id
    minion.name = fresh.name
    minion.tier = fresh.tier
    minion.race = fresh.race
    minion.keywords = fresh.keywords
    minion.granted_keywords = frozenset()
    minion.temp_keywords = frozenset()
    minion.abilities = fresh.abilities
    minion.is_golden = fresh.is_golden
    minion.is_token = fresh.is_token
    minion.dbf_id = fresh.dbf_id
    minion.base_attack, minion.base_health = attack, health
    minion.bonus_attack = minion.bonus_health = 0
    minion.temp_attack = minion.temp_health = 0
    minion.count_bonus_granted = (0, 0)
    minion.self_counted = False
    minion.blood_gem_attack = minion.blood_gem_health = 0
    minion.magnetized_count = 0
    minion.magnet_attack = minion.magnet_health = 0
    minion.magnet_abilities = ()
    minion.has_shield = Keyword.SHIELD in minion.all_keywords
    return True


def apply_transform_into_shop_minion(
    player: PlayerState,
    board_idx: int,
    shop_slot: int,
    *,
    patch: PatchContext,
    copy_golden: bool = False,
) -> None:
    if not (0 <= board_idx < len(player.board)):
        raise ValueError(f"invalid transform board index: {board_idx}")
    offer = player.shop[shop_slot]
    if offer is None:
        raise ValueError(f"empty shop slot for transform: {shop_slot}")
    if copy_golden:
        from src.bg_recruitment.triples import make_forged_golden_minion

        player.board[board_idx] = make_forged_golden_minion(
            offer.card_id, patch=patch
        )
    else:
        player.board[board_idx] = make_minion(offer.card_id, patch=patch)


def try_open_transform_shop_modal(
    player: PlayerState,
    board_idx: int,
    *,
    patch: PatchContext,
    rng: np.random.Generator,
    copy_golden: bool = False,
) -> bool:
    """Apply transform or open a shop-slot modal. Returns True if modal opened."""
    slots = filled_shop_slot_indices(player)
    if not slots:
        return False
    if len(slots) == 1:
        apply_transform_into_shop_minion(
            player, board_idx, slots[0], patch=patch, copy_golden=copy_golden
        )
        return False
    opts: list[str] = ["", "", ""]
    for i, slot in enumerate(slots[:3]):
        offer = player.shop[slot]
        assert offer is not None
        opts[i] = offer.card_id
    player.pending_choice = PendingChoice(
        PendingChoiceKind.TRANSFORM_SHOP_MINION,
        (opts[0], opts[1], opts[2]),
        0,
        transform_board_idx=board_idx,
    )
    return True


def resolve_transform_shop_pick(
    player: PlayerState,
    shop_slot: int,
    *,
    patch: PatchContext,
    on_after_placed: Optional[Callable[[PlayerState, object], None]] = None,
) -> None:
    pc = player.pending_choice
    assert pc is not None
    assert pc.kind == PendingChoiceKind.TRANSFORM_SHOP_MINION
    assert pc.transform_board_idx is not None
    board_idx = pc.transform_board_idx
    copy_golden = False
    if 0 <= board_idx < len(player.board):
        board_minion = player.board[board_idx]
        if board_minion is not None:
            from src.bg_core.effects import TransformIntoShopMinionEffect

            for ab in board_minion.abilities:
                if isinstance(ab.effect, TransformIntoShopMinionEffect):
                    copy_golden = ab.effect.copy_golden
                    break
    apply_transform_into_shop_minion(
        player, board_idx, shop_slot, patch=patch, copy_golden=copy_golden
    )
    player.pending_choice = None
    if on_after_placed is not None and 0 <= board_idx < len(player.board):
        on_after_placed(player, player.board[board_idx])
    player.placed_minion_pending_after = None
    player.placed_minion_board_index = None


__all__ = [
    "apply_transform_into_shop_minion",
    "filled_shop_slot_indices",
    "resolve_transform_shop_pick",
    "try_open_transform_shop_modal",
]
