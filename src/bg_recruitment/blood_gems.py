"""Blood Gems — the Quilboar currency: a 0-cost spell that grows a minion.

Everything that hands out, plays or strengthens a Gem goes through here, for
one reason: a Gem is worth ``+1/+1`` *plus whatever the seat has accumulated*
("Your Blood Gems give an extra +1/+1 this game"), and there are two ways to
play one — off your hand onto a chosen minion, or a minion playing one itself
("This plays a Blood Gem on all your other minions"). Two code paths that each
compute the value would drift the first time a card raised it.

What a Gem does, in order:

1. adds the seat's current Gem value to the target's stats;
2. records the same amount on the target, because cards read it back (see
   ``Minion.blood_gem_attack``);
3. hands the target a keyword if the printing grants one *and* the target is a
   Quilboar ("Give a minion +1/+1. If it's a Quilboar, also give it Taunt").

Deliberately not here yet, each needing a system that does not exist:
listeners for "after a Blood Gem is played on this" (Tough Tusk, Geomagus
Roogug), permanent Gems played during combat (they need a channel back out of
the battle), and the Duos "your teammate's minions" targets.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from src.bg_core.board_helpers import fire_spell_cast_on
from src.bg_core.effects import BloodGemTarget, Keyword
from src.bg_core.board_helpers import minion_matches_tribe
from src.bg_core.minion import Minion, Race
from src.bg_core.spell_card import SpellCard
from src.bg_lobby.player import PlayerState

from .hand_slots import first_free_hand_slot
from .shop_auras import refresh_attack_thresholds

__all__ = [
    "BLOOD_GEM_CARD_ID",
    "blood_gem_targets",
    "can_play_blood_gem",
    "blood_gem_value",
    "make_blood_gem",
    "give_blood_gems",
    "play_blood_gem_on",
    "play_blood_gem_from_hand",
    "is_blood_gem",
]

#: The plain Gem. The keyword-granting printings (BG20_GEM_Taunt / _Reborn /
#: _DivineShield) are the same card with ``blood_gem_quilboar_keyword`` set.
BLOOD_GEM_CARD_ID = "BG20_GEM"

_BASE_ATTACK = 1
_BASE_HEALTH = 1


def is_blood_gem(card) -> bool:
    return isinstance(card, SpellCard) and card.is_blood_gem


def blood_gem_value(player: PlayerState) -> Tuple[int, int]:
    """What one Gem is worth to this seat right now: printed +1/+1 plus bonus."""
    return (
        _BASE_ATTACK + player.blood_gem_bonus_attack,
        _BASE_HEALTH + player.blood_gem_bonus_health,
    )


def make_blood_gem(quilboar_keyword: Optional[Keyword] = None) -> SpellCard:
    return SpellCard(
        card_id=BLOOD_GEM_CARD_ID,
        name="Blood Gem",
        cost=0,
        is_tavern_spell=False,  # never offered in the tavern — see SpellCard
        is_blood_gem=True,
        blood_gem_quilboar_keyword=quilboar_keyword,
    )


def give_blood_gems(
    player: PlayerState,
    count: int = 1,
    *,
    quilboar_keyword: Optional[Keyword] = None,
) -> int:
    """Put ``count`` Gems in hand; return how many actually fit.

    A hand with no free slot loses the Gem, the way a full hand loses any other
    card the game tries to hand you.

    Unverified against the client, and the two halves of the question are
    different: that Gems *occupy* hand slots and sit there unplayable when the
    board is empty is documented behaviour (see ``can_play_blood_gem``); what
    the client does with a Gem it cannot fit is not, and no source found so far
    states it. A package that leans on overflow should confirm it first.
    """
    made = 0
    for _ in range(max(0, int(count))):
        slot = first_free_hand_slot(player)
        if slot is None:
            break
        player.hand[slot] = make_blood_gem(quilboar_keyword)
        made += 1
    return made


def play_blood_gem_on(
    player: PlayerState,
    target: Minion,
    *,
    count: int = 1,
    quilboar_keyword: Optional[Keyword] = None,
    patch=None,
) -> None:
    """Play ``count`` Gems onto one minion (from hand or from a card's effect)."""
    if target.cannot_gain_stats:
        return
    attack, health = blood_gem_value(player)
    for _ in range(max(0, int(count))):
        target.bonus_attack += attack
        target.bonus_health += health
        target.blood_gem_attack += attack
        target.blood_gem_health += health
        if quilboar_keyword is not None and target.race is Race.QUILBOAR:
            target.granted_keywords = target.granted_keywords | {quilboar_keyword}
        # A Gem is a spell, and it was cast at this body.
        fire_spell_cast_on(target, player=player, patch=patch)
    refresh_attack_thresholds(player.board)
    # A Gem is a spell, and "spells you've cast" counts every kind.
    from .game_counts import SPELLS_CAST, bump_seat_counter

    for _ in range(max(0, int(count))):
        bump_seat_counter(player, SPELLS_CAST, patch=patch)


def can_play_blood_gem(player: PlayerState) -> bool:
    """Whether a Gem in hand can be played at all — i.e. is there a target.

    A Gem with no minion to land on is not discarded and does not fizzle: it
    stays in hand, unplayable, and keeps taking up the slot. That is a real
    Battlegrounds state, reported often enough to have its own bug threads — a
    hand full of Gems and an empty board leaves a seat with nothing it can do.
    Legality belongs here rather than in the play call, which raises.
    """
    return bool(player.board)


def play_blood_gem_from_hand(
    player: PlayerState,
    hand_index: int,
    board_index: int,
) -> None:
    """Play the Gem in ``hand_index`` onto the board minion at ``board_index``.

    The engine-side half of playing a Gem. Choosing the target is the caller's
    business — the flat action space has no Gem action yet, and giving it one
    moves numbers the trained checkpoints are wired to.
    """
    card = player.hand[hand_index] if 0 <= hand_index < len(player.hand) else None
    if not is_blood_gem(card):
        raise ValueError(f"hand slot {hand_index} does not hold a Blood Gem: {card!r}")
    if not 0 <= board_index < len(player.board):
        raise ValueError(f"no minion at board index {board_index} to receive the Gem")
    play_blood_gem_on(
        player,
        player.board[board_index],
        quilboar_keyword=card.blood_gem_quilboar_keyword,
    )
    player.hand[hand_index] = None


def blood_gem_targets(
    player: PlayerState,
    source: Minion,
    target: BloodGemTarget,
) -> List[Minion]:
    """The board minions a "this plays a Blood Gem on ..." effect reaches."""
    board = player.board
    if target is BloodGemTarget.SELF:
        return [source] if source in board else []
    if target is BloodGemTarget.ALL_FRIENDLY:
        return list(board)
    if target is BloodGemTarget.ALL_OTHER_FRIENDLY:
        return [m for m in board if m is not source]
    if target is BloodGemTarget.ALL_FRIENDLY_QUILBOAR:
        # An Amalgam is a Quilboar too, and "all your **other** Quilboar" leaves
        # the card saying it out — which is what every printing of this says.
        return [
            m
            for m in board
            if m is not source and minion_matches_tribe(m, Race.QUILBOAR)
        ]
    if target is BloodGemTarget.ADJACENT:
        try:
            idx = board.index(source)
        except ValueError:
            return []
        return [board[j] for j in (idx - 1, idx + 1) if 0 <= j < len(board)]
    raise NotImplementedError(f"no target resolution for {target!r}")


def steal_blood_gems(thief: Minion, victims: List[Minion]) -> None:
    """Move every Gem stat off ``victims`` onto ``thief`` (Gem Confiscation).

    Included with the core because it is the second reader of the per-minion
    record, and the pair of them is why that record exists at all.
    """
    for victim in victims:
        attack, health = victim.blood_gem_attack, victim.blood_gem_health
        if not attack and not health:
            continue
        victim.bonus_attack -= attack
        victim.bonus_health -= health
        victim.blood_gem_attack = 0
        victim.blood_gem_health = 0
        thief.bonus_attack += attack
        thief.bonus_health += health
        thief.blood_gem_attack += attack
        thief.blood_gem_health += health
