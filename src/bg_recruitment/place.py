"""Play from hand: PLAY and Magnetic."""

from __future__ import annotations

from typing import Optional, Tuple

from src.bg_core.board_helpers import merge_magnet_abilities
from src.bg_core.effects import Keyword, Trigger
from src.bg_core.minion import Minion, Race
from src.bg_lobby.player import PlayerState

from .game_counts import bump_played
from .pool_ledger import on_sell_minion
from .shop_triggers import ShopTriggers
from .triples import (
    flush_triple_reward_queue_if_idle,
    is_triple_reward_discover_spell,
    play_triple_reward_discover_spell_from_hand,
    resolve_triples_loop,
)


def is_mech(m: Minion) -> bool:
    return m.race in (Race.MECHANICAL, Race.ALL)


def hand_minion_can_magnetize(m) -> bool:
    # ``m`` may be a SpellCard (see PlayerState.hand's HandCard union) — a
    # spell obviously can't magnetize; the isinstance check is what actually
    # guards it, since SpellCard has neither ``all_keywords`` nor ``race``.
    return isinstance(m, Minion) and Keyword.MAGNETIC in m.all_keywords and is_mech(m)


def magnet_target_races(magnet: Minion) -> Tuple[Race, ...]:
    """Which tribes this magnet may attach to.

    Mechs unless the card says otherwise — Prosthetic Hand is printed "Can
    Magnetize to Mechs or Undead", and says so through its binding rather than
    through a special case here.
    """
    from src.bg_core.effects import MagnetizesToTribesEffect

    for ability in magnet.abilities:
        if isinstance(ability.effect, MagnetizesToTribesEffect):
            return tuple(ability.effect.tribes)
    return (Race.MECHANICAL,)


def can_magnetize_onto(magnet: Minion, target: Minion) -> bool:
    """Whether ``magnet`` may attach to ``target``."""
    races = magnet_target_races(magnet)
    return target.race in races or target.race == Race.ALL


def merge_magnetic_inplace(target: Minion, magnet: Minion) -> None:
    """HS-style magnetic: keep target identity/buffs; add magnet stats/keywords/DRs."""
    attack, health = magnet.raw_attack, magnet.max_health
    target.base_attack += attack
    target.base_health += health
    combined_kw = (
        target.keywords
        | target.granted_keywords
        | magnet.keywords
        | magnet.granted_keywords
    ) - {Keyword.MAGNETIC}
    target.keywords = combined_kw
    target.granted_keywords = frozenset()
    target.has_shield = target.has_shield or magnet.has_shield

    added = tuple(magnet.abilities)
    target.abilities = merge_magnet_abilities(target.abilities, added)
    # What the part contributed, kept apart from the printed card so a triple
    # can carry it over. The stats above are folded in rather than derived
    # from this; this is the record, not the source.
    target.magnet_attack += attack
    target.magnet_health += health
    target.magnet_abilities = target.magnet_abilities + added


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
            patch=triggers._patch,
        )
        flush_triple_reward_queue_if_idle(
            player, shop_excluded_race, rng=rng, patch=triggers._patch
        )
        return

    assert len(player.board) < board_size
    player.hand[hand_slot] = None
    if insert_at is None:
        player.board.append(minion)
    else:
        pos = max(0, min(int(insert_at), len(player.board)))
        player.board.insert(pos, minion)
    bump_played(player, minion)
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
        resolve_triples_loop(player, patch=triggers._patch)
        flush_triple_reward_queue_if_idle(
            player, shop_excluded_race, rng=rng, patch=triggers._patch
        )


def magnetize(
    player: PlayerState,
    target: Minion,
    magnet: Minion,
    *,
    triggers: Optional[ShopTriggers] = None,
    echo: bool = True,
) -> None:
    """Attach ``magnet`` to ``target``, and tell everything that watches.

    The one place a Magnetization happens, so the four cards that care all see
    every one of them however it arrived — out of hand, made on the spot by
    Spark Snapper, or echoed by Polarizing Beatboxer.

    Doubling is spent here rather than by the caller: "the next Magnetization to
    this minion is doubled" is a property of the *target*, and only this knows
    which minion is being magnetized to.
    """
    times = 2 if target.magnet_doubles_next else 1
    target.magnet_doubles_next = False
    for _ in range(times):
        merge_magnetic_inplace(target, magnet)
        target.magnetized_count += 1
    if triggers is not None:
        triggers.fire_magnetized(player, target, magnet)
    if echo:
        _echo_magnetize(player, target, magnet, triggers=triggers)


def _echo_magnetize(
    player: PlayerState,
    target: Minion,
    magnet: Minion,
    *,
    triggers: Optional[ShopTriggers],
) -> None:
    """Polarizing Beatboxer: a Magnetization elsewhere also lands on it.

    ``echo=False`` on the second landing, or two Beatboxers would answer each
    other forever.
    """
    from src.bg_core.effects import EchoMagnetizeEffect

    for other in list(player.board):
        if other is target:
            continue
        if any(isinstance(ab.effect, EchoMagnetizeEffect) for ab in other.abilities):
            magnetize(player, other, magnet, triggers=triggers, echo=False)


def magnet_from_hand(
    player: PlayerState,
    hand_slot: int,
    board_pos: int,
    *,
    patch: PatchContext,
    triggers: Optional[ShopTriggers] = None,
    shared_pool=None,
) -> None:
    magnet = player.hand[hand_slot]
    assert magnet is not None
    assert board_pos < len(player.board)
    target = player.board[board_pos]
    assert can_magnetize_onto(magnet, target)
    assert hand_minion_can_magnetize(magnet)
    player.hand[hand_slot] = None
    # "Magnetizing counts as playing a minion but not summoning a minion" — so
    # the play tallies count it, and ``fire_shop_friendly_summoned`` does not.
    # The board triggers are the seat's own reading and stay out of it: the one
    # card that watches both prints *"Whenever you play **or Magnetize** a
    # Mech"*, which it would not need to say if playing covered it.
    bump_played(player, magnet)
    # Patch 27.0.0.181554: "Magnetic minions now return to the minion pool
    # immediately after being Magnetized." The body is gone into the host, so
    # nothing later will release it — without this every Magnetization deleted
    # a copy from the lobby for good.
    on_sell_minion(shared_pool, magnet)
    magnetize(player, target, magnet, triggers=triggers)
    if player.pending_choice is None:
        resolve_triples_loop(player, patch=patch)
