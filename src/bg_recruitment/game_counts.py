"""Game-long tallies, and the stats a card reads off one.

    "Has +3/+2 for each other Ancestral Automaton you've summoned this game"
    "Has +4/+2 for each friendly Eternal Knight that died this game"
    "Has +8/+8 for each Golden minion you've played this game"
    "Has +2/+1 for each Tavern spell you've cast this game"

One shape: the seat counts an event, and every card carrying the ability
recomputes its stats from that count. The count is the seat's, which is what
"wherever this is" means — a copy in hand reads the same tally as one on the
board.

**Recompute, don't accumulate.** The obvious alternative — buff every copy each
time the event fires — has to answer "which copies were standing at the time",
and gets "for each *other*" wrong in two directions at once: fire it from the
copies already out and N of them fire per arrival, so the bonus grows with the
square of the count; fire it from the newcomer and the ones already out miss
their share. A tally plus a recompute has no such question in it. Each card
knows only what it has been granted so far, so the recompute is a difference and
is safe to run at any time.

"Other" is one subtraction: a copy whose own arrival was counted does not count
itself.
"""

from __future__ import annotations

from typing import Dict, Optional

from src.bg_core.effects import SelfBonusPerGameCount, Trigger
from src.bg_core.minion import Minion, is_locked
from src.bg_lobby.player import PlayerState

__all__ = [
    "SUMMONED",
    "DIED",
    "DEATHS",
    "counter_key",
    "GOLDEN_PLAYED",
    "SPELLS_CAST",
    "TAVERN_SPELLS_CAST",
    "DEATHRATTLES_TRIGGERED",
    "bump_died",
    "bump_game_count",
    "bump_played",
    "bump_seat_counter",
    "improve_level",
    "refresh_count_bonuses",
]

#: Counter families. A key is ``"<family>:<subject>"``, where the subject is a
#: card id for the counters a card keeps about itself and ``"*"`` for the
#: seat-wide ones ("each Golden minion you've played").
SUMMONED = "summoned"
DIED = "died"

#: Every spell this seat has cast — Tavern spells, Spellcraft spells and Blood
#: Gems alike, because "spells you've cast" draws no distinction between them.
SPELLS_CAST = "spells_cast:*"

#: Tavern spells only — narrower than SPELLS_CAST, because the cards that read
#: it say "Tavern spell" and a Blood Gem is not one.
TAVERN_SPELLS_CAST = "tavern_spells_cast:*"

#: Deathrattles this seat has triggered, all game. One per firing rather than
#: per body: a deathrattle doubled by Baron is two triggers, and one fired
#: without a death (Deathstrider re-triggering the left-most) is one more.
DEATHRATTLES_TRIGGERED = "deathrattles_triggered:*"

#: Every friendly this seat has lost, whatever it was. The tally beside it is
#: keyed by card id because the cards that read one name a card ("for each
#: Eternal Knight that died"); this one is the plain total, which is what a
#: countdown wants.
DEATHS = "died:*"

#: Golden minions this seat has played. Played, not summoned: the cards that
#: read it say "you've played", and a golden token summoned in a fight is not
#: something the seat played.
GOLDEN_PLAYED = "golden_played:*"


def counter_key(family: str, subject: str = "*") -> str:
    return f"{family}:{subject}"


def improve_level(player: PlayerState, counter: str, per: int = 1) -> int:
    """How many times over a card that "improves" is worth its printed value.

    One to start with — an unimproved card is exactly what it prints — and one
    more per ``per`` events counted. "Improve your future Ballers" counts every
    sale, so ``per`` is 1; "improved by every 4 spells you've cast this game"
    counts every spell and divides.
    """
    if not counter:
        return 1
    return 1 + player.game_counts.get(counter, 0) // max(1, int(per))


def bump_seat_counter(player: PlayerState, counter: str, *, patch=None) -> None:
    """Count one event on a named tally, and let the readers catch up.

    A spell also wakes the board watchers that answer every Nth spell — a
    different question from the seat tally, and asked at the same moment.
    """
    player.game_counts[counter] = player.game_counts.get(counter, 0) + 1
    refresh_count_bonuses(player)
    if counter == SPELLS_CAST and patch is not None:
        from src.bg_core.board_helpers import seat_rng

        from .shop_triggers import ShopTriggers

        ShopTriggers(seat_rng(player), patch=patch).fire_spell_cast(player)


def _abilities(minion: Minion):
    for ability in minion.abilities:
        if ability.trigger is Trigger.AURA and isinstance(
            ability.effect, SelfBonusPerGameCount
        ):
            yield ability.effect


def _key_for(minion: Minion, effect: SelfBonusPerGameCount) -> str:
    """The counter this card reads — its own card id unless the card names one."""
    return counter_key(effect.counter, effect.subject or minion.card_id)


def _refresh_one(player: PlayerState, minion: Minion) -> None:
    owed_attack = owed_health = 0
    for effect in _abilities(minion):
        count = player.game_counts.get(_key_for(minion, effect), 0)
        if not effect.count_self and minion.self_counted:
            count -= 1
        count = max(0, count)
        owed_attack += effect.attack_per * count
        owed_health += effect.health_per * count
    had_attack, had_health = minion.count_bonus_granted
    if (owed_attack, owed_health) == (had_attack, had_health):
        return
    minion.bonus_attack += owed_attack - had_attack
    minion.bonus_health += owed_health - had_health
    minion.count_bonus_granted = (owed_attack, owed_health)


def refresh_count_bonuses(player: PlayerState) -> None:
    """Recompute every count-scaled card the seat owns, wherever it is.

    Idempotent, because each card holds what it has already been granted and
    this applies the difference.
    """
    if not player.game_counts:
        return
    for minion in player.board:
        _refresh_one(player, minion)
    for card in player.hand:
        if isinstance(card, Minion) and not is_locked(card):
            _refresh_one(player, card)
    for card in player.shop:
        if isinstance(card, Minion):
            _refresh_one(player, card)


def bump_game_count(
    player: PlayerState,
    family: str,
    subject: str = "*",
    *,
    subject_card: Optional[Minion] = None,
) -> None:
    """Count one event, then let every card that reads the tally catch up.

    ``subject_card`` is the card the event happened *to*; if it is the one
    keeping this tally, it is marked as counted so that "for each other" leaves
    it out of its own count.
    """
    key = counter_key(family, subject)
    player.game_counts[key] = player.game_counts.get(key, 0) + 1
    if subject_card is not None and any(
        _key_for(subject_card, effect) == key for effect in _abilities(subject_card)
    ):
        subject_card.self_counted = True
    refresh_count_bonuses(player)


def bump_died(player: PlayerState, dead: Minion) -> None:
    """A friendly died — the shop's half of what combat already counts.

    Selling is not this, and neither is a triple merge or being eaten: those are
    the card leaving, not dying. Destroying one is, which is why this exists at
    all — until a card destroyed a friendly in the tavern, every death in the
    game happened in a fight.
    """
    bump_game_count(player, DIED, dead.card_id, subject_card=dead)
    player.game_counts[DEATHS] = player.game_counts.get(DEATHS, 0) + 1


def bump_played(player: PlayerState, played: Minion) -> None:
    """Count a card the seat played from hand, for the tallies that read it."""
    if played.is_golden:
        bump_seat_counter(player, GOLDEN_PLAYED)


def bump_summoned(player: PlayerState, arrived: Minion) -> None:
    """A minion joined the board — by being played, or summoned by anything.

    Both are "summoned" as the cards use the word, which is why this is called
    from the one place every arrival goes through rather than from the buy path.
    """
    bump_game_count(player, SUMMONED, arrived.card_id, subject_card=arrived)
