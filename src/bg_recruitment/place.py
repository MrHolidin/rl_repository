"""Play from hand: PLACE and Magnetic."""

from __future__ import annotations

from typing import Optional

from src.bg_core.effects import Keyword, Trigger
from src.bg_core.minion import Minion, Race
from src.envs.minibg.state import PlayerState

from .shop_triggers import ShopTriggers
from .triples import (
    flush_triple_reward_queue_if_idle,
    resolve_triples_loop,
    try_open_triple_reward_discover,
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
) -> None:
    minion = player.hand[hand_slot]
    assert minion is not None
    assert len(player.board) < board_size
    queued_triple_reward = minion.is_golden and minion.from_triple_merge
    player.hand[hand_slot] = None
    player.board.append(minion)
    triggers.fire_shop_friendly_summoned(player, minion)
    player.placed_minion_board_index = len(player.board) - 1
    player.placed_minion_pending_after = minion
    triggers.fire_on_place(minion, player, shop_excluded_race)
    if player.pending_choice is None:
        try:
            idx = player.board.index(minion)
        except ValueError:
            pass
        else:
            triggers.fire_after_friendly_minion_placed(player, player.board[idx])
        player.placed_minion_board_index = None
        player.placed_minion_pending_after = None

    resolve_triples_loop(player)

    if queued_triple_reward and minion in player.board and minion.is_golden:
        if player.pending_choice is None:
            try_open_triple_reward_discover(player, shop_excluded_race, rng=rng)
        else:
            player.triple_reward_discover_pending = True
        minion.from_triple_merge = False

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
    resolve_triples_loop(player)
