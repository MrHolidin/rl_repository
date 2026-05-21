"""Play from hand: PLACE and Magnetic."""

from __future__ import annotations

from typing import Optional

from src.bg_core.effects import Keyword, Trigger
from src.bg_core.minion import Minion, Race
from src.bg_lobby.player import PlayerState

from .shop_triggers import ShopTriggers
from .triples import (
    flush_triple_reward_queue_if_idle,
    is_triple_reward_discover_spell,
    play_triple_reward_discover_spell_from_hand,
    resolve_triples_loop,
)


def is_mech(m: Minion) -> bool:
    return m.race in (Race.MECHANICAL, Race.ALL)


def hand_minion_can_magnetize(m: Minion) -> bool:
    return Keyword.MAGNETIC in m.all_keywords and is_mech(m)


def merge_magnetic_inplace(target: Minion, magnet: Minion) -> None:
    """HS-style magnetic: keep target identity/buffs; add magnet stats/keywords/DRs."""
    target.base_attack += magnet.raw_attack
    target.base_health += magnet.max_health
    combined_kw = (
        target.keywords
        | target.granted_keywords
        | magnet.keywords
        | magnet.granted_keywords
    ) - {Keyword.MAGNETIC}
    target.keywords = combined_kw
    target.granted_keywords = frozenset()
    target.has_shield = target.has_shield or magnet.has_shield

    nt = [ab for ab in target.abilities if ab.trigger != Trigger.ON_DEATH]
    dt = [ab for ab in target.abilities if ab.trigger == Trigger.ON_DEATH]
    nm = [ab for ab in magnet.abilities if ab.trigger != Trigger.ON_DEATH]
    dm = [ab for ab in magnet.abilities if ab.trigger == Trigger.ON_DEATH]
    target.abilities = tuple(nt + nm + dt + dm)


def place_from_hand(
    player: PlayerState,
    hand_slot: int,
    shop_excluded_race: Optional[Race],
    *,
    board_size: int,
    triggers: ShopTriggers,
    rng,
    insert_at: Optional[int] = None,
    apply_targeted_effects: bool = True,
    forced_buff_target: Optional[Minion] = None,
    shared_pool=None,
) -> None:
    minion = player.hand[hand_slot]
    assert minion is not None

    if is_triple_reward_discover_spell(minion):
        play_triple_reward_discover_spell_from_hand(
            player,
            hand_slot,
            shop_excluded_race,
            rng=rng,
            shared_pool=shared_pool,
        )
        flush_triple_reward_queue_if_idle(player, shop_excluded_race, rng=rng)
        return

    assert len(player.board) < board_size
    player.hand[hand_slot] = None
    if insert_at is None:
        player.board.append(minion)
    else:
        pos = max(0, min(int(insert_at), len(player.board)))
        player.board.insert(pos, minion)
    triggers.fire_shop_friendly_summoned(player, minion)
    player.placed_minion_board_index = len(player.board) - 1
    player.placed_minion_pending_after = minion
    triggers.fire_on_place(
        minion, player, shop_excluded_race, shared_pool=shared_pool
    )
    if apply_targeted_effects and player.pending_choice is None:
        from src.bg_recruitment.targeted_battlecry import apply_targeted_on_place_battlecries

        apply_targeted_on_place_battlecries(
            triggers,
            player,
            minion,
            rng=rng,
            forced_buff_target=forced_buff_target,
        )
    if player.pending_choice is None and apply_targeted_effects:
        try:
            idx = player.board.index(minion)
        except ValueError:
            pass
        else:
            triggers.fire_after_friendly_minion_placed(player, player.board[idx])
        player.placed_minion_board_index = None
        player.placed_minion_pending_after = None

    # Resolve triples only after discover/adapt modals close — otherwise a merge can
    # fill the hand while pending_choice is still set and soft-lock the shop (no legal actions).
    if player.pending_choice is None:
        resolve_triples_loop(player)
        flush_triple_reward_queue_if_idle(player, shop_excluded_race, rng=rng)


def magnet_from_hand(
    player: PlayerState,
    hand_slot: int,
    board_pos: int,
) -> None:
    magnet = player.hand[hand_slot]
    assert magnet is not None
    assert board_pos < len(player.board)
    target = player.board[board_pos]
    assert is_mech(target)
    assert hand_minion_can_magnetize(magnet)
    player.hand[hand_slot] = None
    merge_magnetic_inplace(target, magnet)
    if player.pending_choice is None:
        resolve_triples_loop(player)
