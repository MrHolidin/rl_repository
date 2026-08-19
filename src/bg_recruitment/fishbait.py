"""Fishbait — a 0/1 the tavern puts up for your Beast to kill.

Two cards set it up: Lurking Lionfish replaces a tavern card with one
("...for your left-most Beast to attack"), and Snarky Shark refreshes the
tavern with one when sold ("Your left-most Beast attacks it"). The bait is a
2-tier Beast token, 0/1, that cannot gain stats, and whose deathrattle gives
the minion that killed it +5/+5 — +10/+10 from the Golden printing.

**This is an attack, in the recruit phase.** That is the whole difficulty: the
combat engine works on two boards of copies, and nothing here has either. The
narrow reading is safe because the bait is fixed: 0 Attack, so the attacker
never takes damage; 1 Health and no way to gain any, so the swing always kills
it; one deathrattle, whose only argument is the killer. The exchange therefore
has exactly one outcome and needs no combat runtime to find it.

What it does still owe the rest of the engine is **Rally**: "whenever this
attacks" fires here too, because this is an attack. Those abilities go through
the shop dispatcher, which raises on anything it cannot apply in a tavern — a
Rally that deals damage to an enemy minion has no enemy board to look at, and
should be an explicit decision when a package first prints one, not a silent
no-op.

A general "minion dies outside combat" path is deliberately not built here.
When a second card of this shape appears, that is the moment to build it.
"""

from __future__ import annotations

from typing import Callable, Optional

from src.bg_core.effects import Trigger
from src.bg_core.minion import Minion, Race
from src.bg_lobby.player import PlayerState

__all__ = [
    "FISHBAIT_CARD_ID",
    "make_fishbait",
    "leftmost_beast",
    "place_fishbait",
    "attack_fishbait",
    "fire_tavern_rally",
]

FISHBAIT_CARD_ID = "BG36_205"
#: What the bait's deathrattle hands its killer, per printing.
FISHBAIT_REWARD = 5
FISHBAIT_REWARD_GOLDEN = 10


def make_fishbait(*, golden: bool = False) -> Minion:
    return Minion(
        card_id=FISHBAIT_CARD_ID,
        base_attack=0,
        base_health=1,
        tier=2,
        name="Fishbait",
        race=Race.BEAST,
        is_token=True,
        is_golden=golden,
        cannot_gain_stats=True,
    )


def leftmost_beast(
    player: PlayerState, *, exclude: Optional[Minion] = None
) -> Optional[Minion]:
    """The attacker both cards name: left-most Beast on the board.

    ``exclude`` is Snarky Shark, which is a Beast and is still standing there
    when its own sale fires — the swing belongs to a Beast that stays.
    """
    for minion in player.board:
        if minion is exclude:
            continue
        if minion.race in (Race.BEAST, Race.ALL):
            return minion
    return None


def place_fishbait(player: PlayerState, shop_index: int, *, golden: bool = False) -> Minion:
    """Put a bait in a tavern slot, replacing whatever was there."""
    if not 0 <= shop_index < len(player.shop):
        raise ValueError(f"no tavern slot {shop_index}")
    bait = make_fishbait(golden=golden)
    player.shop[shop_index] = bait
    return bait


def attack_fishbait(
    player: PlayerState,
    shop_index: int,
    *,
    fire_rally: Optional[Callable[[Minion], None]] = None,
    exclude: Optional[Minion] = None,
) -> Optional[Minion]:
    """The left-most Beast attacks the bait in ``shop_index``.

    Returns the attacker, or None when the seat has no Beast to send — in which
    case the bait simply stays on the counter, the way a Blood Gem stays in hand
    with nothing to spend it on.
    """
    bait = player.shop[shop_index] if 0 <= shop_index < len(player.shop) else None
    if bait is None or bait.card_id != FISHBAIT_CARD_ID:
        raise ValueError(f"tavern slot {shop_index} does not hold a Fishbait")

    attacker = leftmost_beast(player, exclude=exclude)
    if attacker is None:
        return None

    # An attack is an attack: Rally goes off before the bait dies to it.
    if fire_rally is not None:
        fire_rally(attacker)

    reward = FISHBAIT_REWARD_GOLDEN if bait.is_golden else FISHBAIT_REWARD
    attacker.bonus_attack += reward
    attacker.bonus_health += reward
    player.shop[shop_index] = None
    return attacker


def fire_tavern_rally(player: PlayerState, attacker: Minion, apply_effect) -> None:
    """Fire the attacker's Rally abilities in a tavern attack.

    ``apply_effect(source, effect)`` is the shop dispatcher; it raises on an
    effect with no tavern meaning, which is the intended outcome until someone
    decides what such a Rally should do here.
    """
    for ability in attacker.abilities:
        if ability.trigger is Trigger.ON_ATTACK:
            apply_effect(attacker, ability.effect)
