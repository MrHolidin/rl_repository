"""Board helpers shared by shop and combat effect resolution."""

from __future__ import annotations

import numpy as np

from typing import Any, Optional, Sequence, Tuple

from copy import copy

from .minion import Minion, Race


def count_unique_tribes(
    board: Sequence[Minion],
    *,
    exclude: Optional[Minion] = None,
    exclude_self_card: bool = False,
) -> int:
    """Count distinct non-neutral tribes on ``board`` (``Race.ALL`` ignored).

    ``exclude``: omit this minion instance from the count (Amalgadon self).
    """
    tribes: set[Race] = set()
    for m in board:
        if exclude is not None and m is exclude:
            continue
        if exclude_self_card and exclude is not None and m.card_id == exclude.card_id:
            continue
        if m.race is None or m.race == Race.ALL:
            continue
        tribes.add(m.race)
    return len(tribes)


def minion_matches_tribe(minion: Minion, tribe: Any) -> bool:
    if minion.race is None:
        return False
    if tribe == Race.ALL or minion.race == Race.ALL:
        return True
    return minion.race == tribe


def count_friendly_tribe(
    board: Sequence[Minion],
    tribe: Any,
    *,
    exclude: Optional[Minion] = None,
) -> int:
    return sum(
        1
        for m in board
        if (exclude is None or m is not exclude) and minion_matches_tribe(m, tribe)
    )


def count_golden_friendlies(
    board: Sequence[Minion],
    *,
    exclude: Optional[Minion] = None,
) -> int:
    return sum(
        1 for m in board if (exclude is None or m is not exclude) and m.is_golden
    )


def multiplier_for(minions, kind) -> int:
    """Product of the ``kind`` multiplier auras standing on a board.

    One scan for what used to be four near-identical copies: Brann in the shop,
    Baron and Khadgar in combat, and Khadgar again in the shop once the tavern
    half of his text was implemented. Takes plain ``Minion`` templates so both
    phases can call it -- combat passes ``bm.template``.
    """
    from .effects import Multiplier, Trigger

    p = 1
    for m in minions:
        for ab in m.abilities:
            if ab.trigger == Trigger.AURA and isinstance(ab.effect, Multiplier):
                if ab.effect.kind is kind:
                    p *= ab.effect.factor
    return p


def buff_matching_hits(
    effect: Any,
    candidate: Minion,
    source: Optional[Minion] = None,
    *,
    idx_candidate: Optional[int] = None,
    idx_source: Optional[int] = None,
) -> bool:
    """Does ``candidate`` match ``effect``'s target predicate?

    Operates on the *template* level so shop (``Minion``) and combat
    (``BattleMinion.template``) share one predicate. Aliveness and the
    combat rule "a deathrattle never buffs its own corpse" stay at the call
    sites, which is where they were before the merge.
    """
    from .effects import BuffTarget

    if getattr(effect, "exclude_source", False) and source is not None and candidate is source:
        return False
    t = effect.target
    if t is BuffTarget.ALL_FRIENDLY:
        return True
    if t is BuffTarget.FRIENDLY_OF_TRIBE:
        return minion_matches_tribe(candidate, effect.tribe)
    if t is BuffTarget.OTHER_OF_TRIBE:
        if source is not None and candidate is source:
            return False
        return minion_matches_tribe(candidate, effect.tribe)
    if t is BuffTarget.FRIENDLY_WITH_KEYWORD:
        return effect.keyword in candidate.all_keywords
    if t is BuffTarget.FRIENDLY_GOLDEN:
        return bool(candidate.is_golden)
    if t is BuffTarget.ADJACENT:
        # Positional, so the caller has to supply where the two stand.
        if idx_candidate is None or idx_source is None:
            return False
        return idx_candidate in (idx_source - 1, idx_source + 1)
    raise ValueError(f"unhandled BuffTarget {t!r}")


def index_of(minions: Sequence[Minion], minion: Minion) -> Optional[int]:
    """Slot ``minion`` occupies, or ``None`` if it is not on this board.

    By identity, not by value: two Alleycats are equal in every printed field,
    and the slot is what adjacency and summon anchoring are decided by.
    """
    for i, m in enumerate(minions):
        if m is minion:
            return i
    return None


def stat_aura_bonus(
    minions: Sequence[Minion],
    recipient: Minion,
    *,
    live_only: bool = False,
) -> Tuple[int, int]:
    """Attack/health ``recipient`` is getting from its boardmates' auras.

    Both phases ran this same loop: find the recipient's slot, walk every
    other minion's AURA abilities, and sum the ``StatAura`` ones that reach it.
    The slot matters because ADJACENT (Dire Wolf Alpha) is decided by it.

    ``live_only`` is combat's, and it is load-bearing rather than defensive: a
    minion that has died is not reaped until its death finishes resolving, so
    it is briefly still in the list, and a corpse must not keep buffing. Nothing
    outside combat maintains damage, so no shop minion is ever in that state.
    """
    from .effects import StatAura, Trigger  # circular at module scope

    idx_r = index_of(minions, recipient)
    if idx_r is None:
        return 0, 0
    atk = 0
    hp = 0
    for idx_s, source in enumerate(minions):
        if source is recipient:
            continue
        if live_only and not source.alive:
            continue
        for ab in source.abilities:
            if ab.trigger != Trigger.AURA or not isinstance(ab.effect, StatAura):
                continue
            if buff_matching_hits(
                ab.effect, recipient, idx_candidate=idx_r, idx_source=idx_s
            ):
                atk += ab.effect.attack
                hp += ab.effect.health
    return atk, hp


def fire_spell_cast_on(
    target: Minion,
    *,
    player=None,
    patch=None,
    spell_card_id: str = "",
    spellcraft: bool = False,
) -> None:
    """Fire ``ON_TARGETED_BY_SPELL`` listeners on the minion a spell just hit.

    Both casts the engine can aim at a body — a Spellcraft spell and a Blood Gem
    — come through here, so a card counting them cannot see one kind and miss
    the other. Each Gem of a multi-Gem play is its own cast.

    ``player`` and ``patch`` are needed only by the listeners that write to the
    seat rather than to the body ("give minions in the Tavern +2/+2 this game");
    without them such a card raises rather than silently doing nothing.

    ``spell_card_id`` is for the one listener that answers with the spell
    itself — Zesty Shaker gets another copy of what was cast on it, and cannot
    infer which that was from a buff that has already landed. ``spellcraft``
    says what kind of cast it was, which the caller knows and the card id does
    not: a Spellcraft spell is minted at play time and its catalog row carries
    no such flag.
    """
    import numpy as _np

    def _rng():
        return _np.random.default_rng(0)

    from .effects import (
        BuffOnSpellCastOnTribeEffect,
        BuffSelf,
        CopyTargetingSpellEffect,
        Trigger,
    )

    if player is not None:
        _fire_board_spell_watchers(player, target, patch=patch)
    for ab in target.abilities:
        if ab.trigger is not Trigger.ON_TARGETED_BY_SPELL:
            continue
        eff = ab.effect
        if isinstance(eff, BuffSelf):
            apply_buff_self(target, eff)
        elif isinstance(eff, CopyTargetingSpellEffect):
            _copy_targeting_spell(
                target, eff, player, patch, spell_card_id, spellcraft
            )
        elif player is not None:
            # Anything that writes to the seat rather than to the body goes to
            # the shop dispatcher, which raises on an effect nobody handles.
            from src.bg_recruitment.shop_triggers import ShopTriggers

            ShopTriggers(_rng(), patch=patch).apply_shop_effect(
                player, target, eff, placed=None
            )
        else:
            raise NotImplementedError(
                f"{type(eff).__name__} has no ON_TARGETED_BY_SPELL handler "
                f"(minion {target.card_id})"
            )


def has_attack_threshold_ability(minion: Minion) -> bool:
    """Whether this minion watches its own Attack for a keyword latch.

    Cheap enough to ask once when a minion joins a board, which is what combat
    does — a per-recount scan of every ability would sit in the hot path.
    """
    from .effects import GrantKeywordAtAttackThreshold, Trigger

    return any(
        ab.trigger is Trigger.AURA
        and isinstance(ab.effect, GrantKeywordAtAttackThreshold)
        for ab in minion.abilities
    )


def apply_attack_thresholds(minion: Minion, attack: int) -> bool:
    """Grant any latched keyword this minion's ``attack`` has now earned.

    ``attack`` is passed in rather than read off the minion because the two
    phases measure it differently: combat counts auras and the shop does not.
    Returns whether anything was granted, which combat uses to know the health
    auras need recomputing.
    """
    from .effects import GrantKeywordAtAttackThreshold, Trigger

    granted = False
    for ab in minion.abilities:
        if ab.trigger is not Trigger.AURA:
            continue
        eff = ab.effect
        if not isinstance(eff, GrantKeywordAtAttackThreshold):
            continue
        if attack < eff.threshold or eff.keyword in minion.keywords:
            continue
        grant_keyword(minion, eff.keyword)
        granted = True
    return granted


def _copy_targeting_spell(
    target: Minion, effect, player, patch, spell_card_id: str, spellcraft: bool
) -> None:
    """"When a Spellcraft spell is played on this, get a new copy of it."

    Once per turn on both printings, and the latch is the body's own — the
    same shape ``spellcraft_kept_this_turn`` beside it already has.
    """
    if player is None or patch is None or not spell_card_id:
        return
    if effect.spellcraft_only and not spellcraft:
        return
    if effect.once_per_turn and target.spell_answered_this_turn:
        return
    from src.bg_recruitment.hand_slots import first_free_hand_slot

    spell = patch.tavern_spells.get(spell_card_id)
    if spell is None:
        return
    paid = False
    for _ in range(max(1, effect.count)):
        slot = first_free_hand_slot(player)
        if slot is None:
            break
        player.hand[slot] = spell
        paid = True
    if paid and effect.once_per_turn:
        target.spell_answered_this_turn = True


def _fire_board_spell_watchers(player, target: Minion, *, patch=None) -> None:
    """Board listeners of "whenever you cast a spell on a <tribe>".

    The watcher is not the target, which is what separates this from
    ``ON_TARGETED_BY_SPELL``: Torrential Ruiner pays out for spells cast on any
    friendly Naga, itself included.
    """
    from .effects import BuffOnSpellCastOnTribeEffect, Trigger

    for watcher in list(player.board):
        for ability in watcher.abilities:
            if ability.trigger is not Trigger.AURA:
                continue
            eff = ability.effect
            if not isinstance(eff, BuffOnSpellCastOnTribeEffect):
                continue
            if eff.tribe is not None and not minion_matches_tribe(target, eff.tribe):
                continue
            for friendly in player.board:
                friendly.bonus_attack += eff.attack
                friendly.bonus_health += eff.health
            if eff.effect is not None and patch is not None:
                from src.bg_recruitment.shop_triggers import ShopTriggers

                import numpy as _np

                from .effects import MagnetizeTokenEffect

                triggers = ShopTriggers(_np.random.default_rng(0), patch=patch)
                if isinstance(eff.effect, MagnetizeTokenEffect):
                    # The recipient is the minion the spell was cast on, not the
                    # watcher — and the dispatcher has no way to be told which,
                    # which is why this one is called directly.
                    triggers.apply_magnetize_token(player, target, eff.effect)
                else:
                    triggers.apply_shop_effect(
                        player, watcher, eff.effect, placed=None
                    )


def grant_keyword(minion: Minion, keyword) -> bool:
    """Give ``minion`` a keyword. Returns whether it did not already have it.

    Divine Shield is two facts, not one: the keyword (which a golden copy
    inherits) and ``has_shield`` (whether the shield is up right now, which is
    all that popping it clears). Re-granting the keyword re-arms the shield --
    that is why the second assignment is outside the "was it new" check.

    The return value is combat's: only a keyword that is actually new can
    change what the health auras compute, so only then is it worth marking
    them dirty.
    """
    from .effects import Keyword

    is_new = keyword not in minion.keywords
    if is_new:
        minion.keywords = frozenset(minion.keywords | {keyword})
    if keyword == Keyword.SHIELD:
        minion.has_shield = True
    return is_new


def grant_keyword_random(
    effect,
    minions: Sequence[Minion],
    source: Optional[Minion],
    *,
    rng,
    grant,
) -> None:
    """Give ``effect.keyword`` to a random eligible friendly, ``repeats`` times.

    Selfless Hero in combat and Toxfin in the tavern. The shop honoured
    ``exclude_self``; combat dropped the source unconditionally, which agreed
    with the flag only because every card that exists sets it. Monstrous Macaw
    is what makes the difference reachable at all -- it fires a deathrattle
    while the source is alive and eligible -- so the flag decides now.

    One pool for all repeats: nothing a keyword grant does can change who is
    eligible, and rebuilding it per repeat only obscured that.
    """
    pool = [
        m
        for m in minions
        if not (effect.exclude_self and m is source)
        and (
            effect.filter_race is None
            or minion_matches_tribe(m, effect.filter_race)
        )
    ]
    if not pool:
        return
    for _ in range(max(1, effect.repeats)):
        grant(pool[int(rng.integers(0, len(pool)))], effect.keyword)


def set_minion_stats(minion: Minion, attack: int, health: int) -> None:
    """Set a body's stats outright, discarding what it was.

    Written as bonuses over the printed card rather than by rewriting the base,
    so the minion is still the card it was — a triple still merges it, and the
    catalog still describes it. Temporary buffs are cleared too: they were part
    of the stats being replaced.
    """
    minion.temp_attack = 0
    minion.temp_health = 0
    minion.bonus_attack = int(attack) - minion.base_attack
    minion.bonus_health = int(health) - minion.base_health


def apply_buff_self(minion: Minion, effect, *, grant=None) -> None:
    """"Gain +N/+N", and the keyword some printings pair it with.

    One body because the keyword arrived late: nine sites applied the stats and
    would each have had to learn about it, and the ninth to be missed is a card
    that silently ships half its text.

    ``grant`` is how combat hands out a keyword (it has health auras to mark
    dirty afterwards); the shop has none and passes nothing.
    """
    minion.bonus_attack += effect.attack
    minion.bonus_health += effect.health
    keyword = getattr(effect, "keyword", None)
    if keyword is None:
        return
    if grant is not None:
        grant(minion, keyword)
    else:
        minion.granted_keywords = minion.granted_keywords | {keyword}


def apply_summoned_listener(
    effect,
    listener: Minion,
    summoned: Minion,
    *,
    grant_keyword,
    improve=None,
    in_combat: bool = False,
) -> None:
    """One of the "a friendly minion was summoned" effects.

    They gate on the newcomer's tribe and differ only in what happens next: the
    newcomer gets stats, or double its own, or the listener's; the listener gets
    stats, or a keyword. Both phases wrote out the same dispatch with the same
    tribe check repeated inside each branch.

    ``in_combat`` gates the ones printed "in combat" (Banana Slamma, Stalwart
    Kodo). A shop summon fires the same trigger, and without this they would
    pay there too.

    ``grant_keyword`` is the one part they cannot share: combat has to mark
    its health auras dirty afterwards, and the shop has no auras to mark.
    ``improve`` is the second: a body that improves inside a fight has to write
    the tally through to the owner's minion, and the shop body *is* the owner's.
    """
    from .effects import (
        BuffListenerIfSummonedMatches,
        BuffSummonedIfRace,
        MultiplySummonedAttackEffect,
        GiveOwnStatsToSummonedEffect,
        GrantListenerKeywordIfSummonedMatches,
    )

    if not isinstance(
        effect,
        (
            BuffSummonedIfRace,
            MultiplySummonedAttackEffect,
            GiveOwnStatsToSummonedEffect,
            GrantListenerKeywordIfSummonedMatches,
            BuffListenerIfSummonedMatches,
        ),
    ):
        return
    tribe = getattr(effect, "tribe", None)
    if tribe is not None and not minion_matches_tribe(summoned, tribe):
        return
    if isinstance(effect, BuffSummonedIfRace):
        level = 1 + listener.self_improves if effect.improves else 1
        summoned.bonus_attack += effect.attack * level
        summoned.bonus_health += effect.health * level
        if effect.improves:
            listener.self_improves += 1
            if improve is not None:
                improve(listener)
    elif isinstance(effect, MultiplySummonedAttackEffect):
        if not in_combat:
            return
        current = summoned.raw_attack + summoned.aura_attack
        summoned.bonus_attack += current * max(0, effect.factor - 1)
    elif isinstance(effect, GiveOwnStatsToSummonedEffect):
        if not in_combat:
            return
        if listener.combat_uses_left < 0:
            listener.combat_uses_left = int(effect.charges)
        if listener.combat_uses_left == 0:
            return
        if listener.combat_uses_left > 0:
            listener.combat_uses_left -= 1
        factor = max(1, int(effect.factor))
        summoned.bonus_attack += (listener.raw_attack + listener.aura_attack) * factor
        summoned.bonus_health += listener.max_health * factor
    elif isinstance(effect, GrantListenerKeywordIfSummonedMatches):
        grant_keyword(listener, effect.keyword)
    else:
        listener.bonus_attack += effect.attack
        listener.bonus_health += effect.health


def distinct_tribe_count(minions) -> int:
    """How many different minion types are standing here.

    An Amalgam is every tribe, so one alone answers with all of them — the
    same stance the seat's most-common-tribe count takes, and for the same
    reason: being every tribe is the whole of what the card does.
    """
    from src.bg_core.minion import ALL_TRIBES

    return sum(
        1
        for tribe in ALL_TRIBES
        if any(minion_matches_tribe(m, tribe) for m in minions)
    )


def apply_buff_matching(
    effect, minions, source=None, *, repeats: int = 1, grant=None, rng=None
) -> None:
    """Apply a ``BuffMatching`` to everyone on ``minions`` it reaches.

    One body for what the shop and combat each spelled out. Everything they
    differed by is a parameter now: which minions (a board or a battle side),
    who the source is, and how many times the trigger fires (Baron). Combat
    settles its auras afterwards at the call site, which is the one thing that
    genuinely has no shop equivalent.

    ``limit`` caps how many of the eligible bodies are paid, and ``leftmost``
    says which ones: the first in board order for "your **left-most** Dragon",
    and that many at random for "give **two** friendly Beasts", which is what
    the plain wording means. Without an ``rng`` the random case falls back to
    board order, so a caller with no generator degrades rather than raises.

    ``grant_keyword`` hands a keyword out to everyone reached; ``grant`` is how
    combat injects its own granting function, which marks the health auras
    dirty. Positions are supplied to the predicate now, so the positional
    ``ADJACENT`` target works here too rather than silently matching nobody.

    ``source`` is passed through to the predicate, which excludes it only for
    ``OTHER_OF_TRIBE`` -- the targets whose card text says "other". Combat used
    to exclude the source unconditionally, which is invisible for a real death
    (the body is in the graveyard) but wrong when Monstrous Macaw fires a
    living minion's deathrattle: Goldrinn is a Beast and "give your Beasts"
    includes it.
    """
    limit = int(getattr(effect, "limit", 0) or 0)
    keyword = getattr(effect, "grant_keyword", None)
    give = grant if grant is not None else grant_keyword
    idx_source = index_of(minions, source) if source is not None else None
    leftmost = bool(getattr(effect, "leftmost", False))
    if getattr(effect, "repeat_per_tribe_kind", False):
        kinds = distinct_tribe_count(minions)
        if not kinds:
            return
        repeats = max(1, repeats) * kinds
    for _ in range(max(1, repeats)):
        eligible = [
            m
            for i, m in enumerate(minions)
            if buff_matching_hits(
                effect, m, source, idx_candidate=i, idx_source=idx_source
            )
        ]
        if limit and len(eligible) > limit:
            if leftmost or rng is None:
                eligible = eligible[:limit]
            else:
                picked = rng.choice(len(eligible), size=limit, replace=False)
                chosen = {int(i) for i in np.atleast_1d(picked)}
                eligible = [m for i, m in enumerate(eligible) if i in chosen]
        for m in eligible:
            m.bonus_attack += effect.attack
            m.bonus_health += effect.health
            if keyword is not None:
                give(m, keyword)


def count_for_source(
    source: "CountSource",
    board: Sequence[Minion],
    *,
    tribe: Any = None,
    exclude: Optional[Minion] = None,
) -> int:
    """Dispatch a :class:`CountSource` onto the matching board count."""
    from .effects import CountSource

    if source is CountSource.FRIENDLY_OF_TRIBE:
        return count_friendly_tribe(board, tribe, exclude=exclude)
    if source is CountSource.UNIQUE_TRIBES:
        return count_unique_tribes(board, exclude=exclude)
    if source is CountSource.GOLDEN_FRIENDLIES:
        return count_golden_friendlies(board, exclude=exclude)
    raise ValueError(f"unhandled CountSource {source!r}")


def apply_buff_self_per_count(
    effect: "BuffSelfPerCount",
    listener: Minion,
    board: Sequence[Minion],
) -> None:
    """Apply ``BuffSelfPerCount`` to ``listener`` (its own board is ``board``).

    Single implementation for what used to be three copies of this body, one
    per counting class.
    """
    n = count_for_source(
        effect.source,
        board,
        tribe=effect.tribe,
        exclude=listener if effect.exclude_self else None,
    )
    listener.bonus_attack += effect.attack_per * n
    listener.bonus_health += effect.health_per * n


def snapshot_warband(board: Sequence[Minion]) -> Tuple[Minion, ...]:
    """Shallow-copy minions for ``PlayerState.last_opponent_board``."""
    return tuple(copy(m) for m in board)


__all__ = [
    "apply_buff_self_per_count",
    "apply_buff_matching",
    "distinct_tribe_count",
    "apply_buff_self",
    "set_minion_stats",
    "apply_summoned_listener",
    "apply_attack_thresholds",
    "fire_spell_cast_on",
    "has_attack_threshold_ability",
    "grant_keyword",
    "grant_keyword_random",
    "index_of",
    "stat_aura_bonus",
    "buff_matching_hits",
    "count_for_source",
    "multiplier_for",
    "count_unique_tribes",
    "minion_matches_tribe",
    "count_friendly_tribe",
    "count_golden_friendlies",
    "snapshot_warband",
]
