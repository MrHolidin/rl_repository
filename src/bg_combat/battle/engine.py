"""Combat engine: event dispatch, swing resolution, attacker rotation,
start-of-combat firing, and first-side decision."""
from __future__ import annotations

from typing import List, Optional, Tuple

from src.bg_core.effects import (
    SetEnemyHealthEffect,
    MultiplyFriendlyAttackEffect,
    GainNearestEnemyStatsEffect,
    BuffMatching,
    CastSpellAtEffect,
    DealDamageAllMinions,
    DevourNeighbourEffect,
    GainStatsFromHandEffect,
    Keyword,
    RepeatPerCountEffect,
    SummonBestFromHandEffect,
    StartOfCombatDamagePerFriendlyTribe,
    SummonSelfCopyFromHandEffect,
    Trigger,
)

from src.bg_core.minion import Minion
from .state import BattleMinion, BattleSide, _CombatRuntime, battle_copy
from .summon import _summon_append
from .effects import (
    _deal_damage_to_battle_minion,
    _summon_best_from_hand,
    cast_spell_in_combat,
)
from .events import (
    AttackCompleted,
    BeginAttackExchange,
    DamageDealt,
    DamageStrike,
    MinionDied,
    MinionSummoned,
    Overkill,
    ShieldLost,
)
from .auras import attack_value, _grant_keyword, _sync_health_all
from src.bg_core.board_helpers import apply_buff_matching, minion_matches_tribe
from src.bg_core.minion import ALL_TRIBES
from .targeting import (
    _attacker_has_cleave,
    _cleave_victim_ids_at_swing_start,
    _pick_target,
)
from .effects import (
    _count_friendlies_of_tribe,
    _deal_random_enemy_minion_damage,
    _devour_neighbour,
    _enqueue_strike_events,
    _fire_rally,
    _fire_when_attacked,
    _handle_attack_completed,
    _handle_damage_dealt,
    _handle_minion_died,
    _handle_minion_summoned,
    _handle_overkill,
    _handle_shield_lost,
)


def _queue_hand_start_of_combat(
    rt: _CombatRuntime,
    side_idx: int,
    pending: List[Tuple[BattleMinion, object]],
) -> None:
    """Queue the Start of Combat triggers of cards held in *hand*.

    Flighty Scout is the shape: "If this minion is in your hand, summon a copy
    of it". The card never joins the board, so it is materialised only to carry
    its ability into the queue below — and only for effects that say they fire
    from hand, since every other Start of Combat means the board.

    Queuing locks the card for this fight. It does not leave the hand; it is
    simply not available to be summoned out of again, which is the rule for
    every summon from hand.
    """
    for instance_id, card_id, _attack, _health in rt.seats[side_idx].hand_minions():
        template = rt.patch.templates.get(card_id)
        if template is None:
            continue
        for ab in template.abilities:
            if ab.trigger != Trigger.ON_START_OF_COMBAT:
                continue
            if not isinstance(ab.effect, SummonSelfCopyFromHandEffect):
                continue
            # The card is spent for this fight the moment it queues: it stays in
            # hand, but nothing else may summon it again — the same lock a Rally
            # that reaches into the hand takes.
            rt.hand_summoned[side_idx].add(instance_id)
            pending.append((battle_copy(template, rt.alloc_id()), ab.effect))


def _fire_hero_start_of_combat(rt: _CombatRuntime, side_idx: int) -> None:
    """The hero powers that land before a blow is struck.

    Illidan's ends and Wagtoggle's one-per-type. Read off the seat rather than
    bound as an ability, because a hero has no body on the board to carry one.
    """
    side = rt.side(side_idx)
    living = list(side.iter_living())
    if not living:
        return
    ends = rt.seats[side_idx].start_combat_ends()
    swingers = []
    if ends is not None:
        for bm in {id(living[0]): living[0], id(living[-1]): living[-1]}.values():
            bm.bonus_attack += ends.attack
            bm.bonus_health += ends.health
            swingers.append(bm)
    per_tribe = rt.seats[side_idx].start_combat_one_per_tribe()
    if per_tribe is not None:
        attack, health = per_tribe
        paid: list = []
        for tribe in ALL_TRIBES:
            pool = [
                m
                for m in living
                if minion_matches_tribe(m, tribe)
                and not any(m is already for already in paid)
            ]
            if not pool:
                continue
            pick = pool[int(rt.rng.integers(0, len(pool)))]
            pick.bonus_attack += attack
            pick.bonus_health += health
            paid.append(pick)
    _sync_health_all(rt)
    if ends is not None and ends.attack_immediately:
        # "...and attack immediately" — after both halves have landed, so the
        # stats are on the body before it swings.
        from .effects import _summon_attack_immediately_if_requested

        for bm in swingers:
            _summon_attack_immediately_if_requested(rt, bm, side_idx)
            while rt.queue:
                _dispatch(rt, rt.queue.popleft())


def _queue_seat_start_of_combat(
    rt: _CombatRuntime,
    side_idx: int,
    pending: List[Tuple[BattleMinion, object]],
) -> None:
    """Queue the Start of Combat effects the seat bought with a spell.

    A spell has no body, so one is made to carry the trigger — the same trick
    the hand's own Start of Combat cards get, and for the same reason: it lets
    a promise take its turn in the alternating order rather than being a second
    pass that always goes first. The carrier never joins a board, and the
    handlers that read a source find a card id and nothing else.
    """
    for ability in rt.seats[side_idx].start_combat_promises():
        carrier = Minion(
            card_id=getattr(ability, "card_id", "") or "SPELL",
            base_attack=0,
            base_health=1,
            tier=1,
            abilities=(ability,),
        )
        pending.append((battle_copy(carrier, rt.alloc_id()), ability.effect))


def _fire_start_of_combat(rt: _CombatRuntime) -> None:
    # Hero Start-of-Combat keyword grants to the left-most minion (Al'Akir:
    # Windfury + Divine Shield + Taunt) — applied before minion start-of-combat.
    for side_idx in (0, 1):
        side = rt.side(side_idx)
        if not side.start_combat_keywords:
            continue
        # Al'Akir grants to the left-most living minion only.
        for bm in side.iter_living():
            for kw in side.start_combat_keywords:
                _grant_keyword(rt, side_idx, bm, kw)
            break
    for side_idx in (0, 1):
        _fire_hero_start_of_combat(rt, side_idx)
    # Both sides' triggers in board order, left to right.
    pending: Tuple[List[Tuple[BattleMinion, object]], List[Tuple[BattleMinion, object]]] = ([], [])
    for side_idx in (0, 1):
        for bm in rt.side(side_idx).minions:
            for ab in bm.abilities:
                if ab.trigger == Trigger.ON_START_OF_COMBAT:
                    pending[side_idx].append((bm, ab.effect))
        _queue_hand_start_of_combat(rt, side_idx, pending[side_idx])
        _queue_seat_start_of_combat(rt, side_idx, pending[side_idx])

    # Real BG draws a dominant player at random, then the sides alternate one
    # trigger at a time, each taking its left-most untriggered minion, with
    # deaths (and the deathrattles they set off) resolved between triggers.
    # Firing side 0's whole board first handed the lower seat every start-of-
    # combat race — and the lower seat is always side 0, since pairings are
    # emitted with ``a < b``. The draw is only taken when both sides have
    # something to fire, so boards without a contest keep their RNG stream.
    if pending[0] and pending[1]:
        turn = int(rt.rng.integers(0, 2))
    else:
        turn = 0 if pending[0] else 1
    pos = [0, 0]
    while pos[0] < len(pending[0]) or pos[1] < len(pending[1]):
        if pos[turn] >= len(pending[turn]):
            turn = 1 - turn
            continue
        bm, eff = pending[turn][pos[turn]]
        pos[turn] += 1
        if bm.alive:
            _apply_start_of_combat_effect(rt, turn, bm, eff)
            _sync_health_all(rt)
            while rt.queue:
                _dispatch(rt, rt.queue.popleft())
        turn = 1 - turn
    _sync_health_all(rt)
    while rt.queue:
        ev = rt.queue.popleft()
        _dispatch(rt, ev)
    # The first of the two moments a side can have room: whatever the boards
    # opened with, before a blow is struck.
    for side_idx in (0, 1):
        _fill_combat_space(rt, side_idx)


def _fill_combat_space(rt: _CombatRuntime, side_idx: int) -> None:
    """Spend "when you have space in combat" charges while there is space.

    Asked at the start of a combat and again after each friendly death, which
    are the two moments a side gains any. A charge buys one summon and is spent
    whether or not every body it promised fit — the room is what the card asks
    for, not room for all of them.
    """
    side = rt.side(side_idx)
    _fill_combat_space_from_hero(rt, side_idx)
    while side.alive_count() < rt.combat_board_max:
        effect = rt.seats[side_idx].take_combat_space_summon()
        if effect is None:
            return
        template = rt.patch.templates.get(effect.token_id)
        if template is None:
            continue
        for _ in range(max(1, int(effect.count))):
            summoned = _summon_append(rt, side_idx, template)
            if summoned is None:
                break
            if effect.grant_keyword is not None:
                _grant_keyword(rt, side_idx, summoned, effect.grant_keyword)
        _sync_health_all(rt)
        while rt.queue:
            _dispatch(rt, rt.queue.popleft())


def _apply_start_of_combat_effect(
    rt: _CombatRuntime, side_idx: int, source: BattleMinion, eff: object
) -> None:
    """One Start-of-Combat trigger. Counts are read now, not up front, so a
    minion killed by an earlier trigger no longer feeds this one's tally."""
    if isinstance(eff, StartOfCombatDamagePerFriendlyTribe):
        count = _count_friendlies_of_tribe(rt.side(side_idx), eff.tribe)
        if count <= 0:
            return
        amount = count * eff.amount_per_match
        for _ in range(max(1, eff.repeats)):
            _deal_random_enemy_minion_damage(rt, side_idx, amount)
    elif isinstance(eff, BuffMatching):
        if eff.lasting:
            # Stays open for the fight, so everything summoned after this is
            # paid too — the buff below still lands on whoever is here now.
            rt.lasting_buffs[side_idx].append(eff)
        # "Start of Combat: give your other Dragons +1/+1", "…your Beasts have
        # +1 Attack for the rest of this combat", "give your left-most Dragon
        # +1/+2 and Windfury" (that one is ``limit=1`` plus a granted keyword).
        # All the same write: it lands on the combat copies and dies with them.
        apply_buff_matching(
            eff,
            rt.side(side_idx).minions,
            source,
            grant=lambda m, kw: _grant_keyword(rt, side_idx, m, kw),
            rng=rt.rng,
        )
        _sync_health_all(rt)
    elif isinstance(eff, DealDamageAllMinions):
        # "Start of Combat: deal 3 damage to all other minions" — every body in
        # the fight but the one that said so. The Golden says "twice", which is
        # two passes and not one of double size: a Divine Shield eats one of
        # them and the second still lands.
        for _ in range(max(1, eff.repeats)):
            for other_side in (0, 1):
                for bm in list(rt.side(other_side).iter_living()):
                    if bm is source:
                        continue
                    _deal_damage_to_battle_minion(rt, other_side, bm, eff.amount)
    elif isinstance(eff, RepeatPerCountEffect):
        # "Improves permanently after you cast a Tavern spell" — the tally is
        # the seat's, so the level is asked of it and the inner effect repeats.
        times = rt.seats[side_idx].improve_level(eff.counter, eff.per)
        for _ in range(max(1, times)):
            _apply_start_of_combat_effect(rt, side_idx, source, eff.effect)
    elif isinstance(eff, DevourNeighbourEffect):
        _devour_neighbour(rt, side_idx, source, eff)
    elif isinstance(eff, CastSpellAtEffect):
        cast_spell_in_combat(rt, side_idx, source, eff.card_id)
    elif isinstance(eff, GainStatsFromHandEffect):
        held = rt.seats[side_idx].hand_minions()
        if held:
            # "**twice**" on the Golden printing of both: the same hand, read
            # again, rather than a deeper reach into it.
            times = max(1, int(getattr(eff, "times", 1)))
            if eff.highest_attack_only:
                source.bonus_attack += max(row[2] for row in held) * times
            else:
                source.bonus_attack += sum(row[2] for row in held) * times
                source.bonus_health += sum(row[3] for row in held) * times
            _sync_health_all(rt)
    elif isinstance(eff, SummonBestFromHandEffect):
        # Same reach as the Rally that summons from hand, at a different moment:
        # the card stays put and a copy joins the fight.
        _summon_best_from_hand(rt, side_idx, source, eff)
    elif isinstance(eff, SummonSelfCopyFromHandEffect):
        # ``source`` is the card in hand, made only to carry this trigger; the
        # copy that joins the fight is built from its template like any summon.
        template = rt.patch.templates.get(source.card_id)
        if template is not None:
            _summon_append(rt, side_idx, template)
    elif isinstance(eff, SetEnemyHealthEffect):
        # Written, not dealt: no Divine Shield eats it and nothing is damaged,
        # so a body set to 1 is a body at full Health with one of it.
        foes = list(rt.side(1 - side_idx).iter_living())
        for _ in range(max(1, eff.count)):
            if not foes:
                break
            victim = foes.pop(int(rt.rng.integers(0, len(foes))))
            victim.bonus_health += int(eff.health) - victim.max_health
        _sync_health_all(rt)
    elif isinstance(eff, MultiplyFriendlyAttackEffect):
        target = _end_minion(rt.side(side_idx), leftmost=eff.leftmost)
        if target is not None:
            # The Attack it is standing there with, auras included -- doubling
            # what a Dire Wolf is lending is what the board shows.
            had = _combat_attack(rt, side_idx, target)
            target.bonus_attack += had * (max(1, eff.factor) - 1)
    elif isinstance(eff, GainNearestEnemyStatsEffect):
        target = _end_minion(rt.side(side_idx), leftmost=eff.leftmost)
        foe = _nearest_enemy(rt, side_idx, target)
        if target is not None and foe is not None:
            target.bonus_attack += _combat_attack(rt, 1 - side_idx, foe)
            target.bonus_health += foe.max_health
            _sync_health_all(rt)
    else:
        raise NotImplementedError(
            f"Start of Combat effect {type(eff).__name__} has no combat handler "
            f"(minion {source.card_id})"
        )


def _fill_combat_space_from_hero(rt: _CombatRuntime, side_idx: int) -> None:
    """"When you have space in combat, summon a copy of your biggest minion."

    Once per fight. The card prints no charge count, unlike Boon of Beetles'
    "(2 left!)", and firing it after every death would refill the board for
    free all combat — so the bound is one, and it is a reading rather than a
    rule the card states.
    """
    if rt.hero_space_summoned[side_idx]:
        return
    by = rt.seats[side_idx].space_summon_copy()
    if by is None:
        return
    side = rt.side(side_idx)
    if side.alive_count() >= rt.combat_board_max:
        return
    living = list(side.iter_living())
    if not living:
        return
    biggest = max(
        living,
        key=(
            (lambda m: m.max_health)
            if by == "health"
            else (lambda m: _combat_attack(rt, side_idx, m))
        ),
    )
    rt.hero_space_summoned[side_idx] = True
    if _summon_append(rt, side_idx, biggest) is not None:
        _sync_health_all(rt)
        while rt.queue:
            _dispatch(rt, rt.queue.popleft())


def _combat_attack(rt: _CombatRuntime, side_idx: int, bm: BattleMinion) -> int:
    """What a body's Attack reads on the board right now, auras included."""
    return attack_value(
        bm,
        rt.side(side_idx),
        death_resolution=rt.in_death_resolution,
        battle_field=rt.sides,
    )


def _end_minion(side: BattleSide, *, leftmost: bool) -> Optional[BattleMinion]:
    living = list(side.iter_living())
    if not living:
        return None
    return living[0] if leftmost else living[-1]


def _nearest_enemy(
    rt: _CombatRuntime, side_idx: int, of: Optional[BattleMinion]
) -> Optional[BattleMinion]:
    """The enemy standing opposite ``of`` — who it would meet.

    The same slot on the other side, and the closest body still standing when
    that slot is empty, which is what "nearest" has to mean once the boards are
    different lengths.
    """
    if of is None:
        return None
    foes = list(rt.side(1 - side_idx).iter_living())
    if not foes:
        return None
    mine = list(rt.side(side_idx).iter_living())
    try:
        idx = mine.index(of)
    except ValueError:
        idx = 0
    return foes[min(idx, len(foes) - 1)]


def _dispatch(rt: _CombatRuntime, event: BattleEvent) -> None:
    if isinstance(event, BeginAttackExchange):
        return
    if isinstance(event, DamageStrike):
        _enqueue_strike_events(rt, event)
        return
    if isinstance(event, ShieldLost):
        _handle_shield_lost(rt, event)
        return
    if isinstance(event, DamageDealt):
        _handle_damage_dealt(rt, event)
        return
    if isinstance(event, Overkill):
        _handle_overkill(rt, event)
        return
    if isinstance(event, AttackCompleted):
        _handle_attack_completed(rt, event)
        return
    if isinstance(event, MinionDied):
        _handle_minion_died(rt, event)
        return
    if isinstance(event, MinionSummoned):
        _handle_minion_summoned(rt, event)
        return


def _run_single_swing(
    rt: _CombatRuntime,
    attacker: BattleMinion,
    target: BattleMinion,
    attacker_side_idx: int,
    defender_side_idx: int,
) -> None:
    atk_side = rt.side(attacker_side_idx)
    def_side = rt.side(defender_side_idx)
    rt.in_death_resolution = False
    if not attacker.alive or not target.alive:
        return
    # One of the owner's minions is swinging, for the heroes that count them.
    rt.seats[attacker_side_idx].count_attack()
    _fire_when_attacked(rt, defender_side_idx, target)
    # Rally reads the board the attack was declared on: after the target is
    # locked in and the defender's own on-attacked triggers have run, but before
    # either side's damage is measured below.
    _fire_rally(rt, attacker, attacker_side_idx, target)
    # A Rally that deals damage can empty the slot it was aiming at, and the
    # deaths it caused are sitting in the queue: drain them here, or they are
    # announced late (or not at all, if this swing turns out to be the last).
    while rt.queue:
        _dispatch(rt, rt.queue.popleft())
    if not attacker.alive or not target.alive:
        return
    bf = (rt.side(0), rt.side(1))
    a_dmg = attack_value(attacker, atk_side, death_resolution=False, battle_field=bf)
    d_dmg = attack_value(target, def_side, death_resolution=False, battle_field=bf)

    rt.swing_damage_survivors.clear()
    rt.attacker_killed_this_swing = False
    rt.queue.append(BeginAttackExchange(attacker_side_idx, defender_side_idx))
    rt.queue.append(
        DamageStrike(
            attacker.instance_id,
            target.instance_id,
            defender_side_idx,
            a_dmg,
        )
    )
    if _attacker_has_cleave(attacker):
        for vid in _cleave_victim_ids_at_swing_start(def_side, target):
            rt.queue.append(
                DamageStrike(
                    attacker.instance_id,
                    vid,
                    defender_side_idx,
                    a_dmg,
                )
            )
    rt.queue.append(
        DamageStrike(
            target.instance_id,
            attacker.instance_id,
            attacker_side_idx,
            d_dmg,
            # The defender's answer, not a swing of its own: it damages and can
            # kill, but it cannot overkill.
            is_attack=False,
        )
    )
    rt.queue.append(
        AttackCompleted(attacker_side_idx, attacker.instance_id)
    )
    while rt.queue:
        ev = rt.queue.popleft()
        _dispatch(rt, ev)


def _run_attacker_activation(
    rt: _CombatRuntime,
    attacker: BattleMinion,
    attacker_side_idx: int,
    defender_side_idx: int,
) -> None:
    """Resolve one board position's attack: Windfury may chain two swings before side swap."""
    attacker_side = rt.side(attacker_side_idx)
    battle_field = (rt.side(0), rt.side(1))
    if not _can_attack(attacker, attacker_side, battle_field=battle_field):
        return
    # Who is swinging, for the cards that are immune only while they are.
    previous_swinger = rt.swinging_instance_id
    rt.swinging_instance_id = attacker.instance_id
    try:
        _run_swings(rt, attacker, attacker_side_idx, defender_side_idx, attacker_side, battle_field)
    finally:
        rt.swinging_instance_id = previous_swinger


def _run_swings(
    rt: _CombatRuntime,
    attacker: BattleMinion,
    attacker_side_idx: int,
    defender_side_idx: int,
    attacker_side,
    battle_field,
) -> None:
    kws = attacker.all_keywords
    if Keyword.MEGA_WINDFURY in kws:
        n_swings = 4
    elif Keyword.WINDFURY in kws:
        n_swings = 2
    else:
        n_swings = 1
    defender_side = rt.side(defender_side_idx)
    for _ in range(n_swings):
        if (
            not _can_attack(attacker, attacker_side, battle_field=battle_field)
            or not defender_side.has_alive()
        ):
            break
        tgt = _pick_target(
            defender_side,
            rt.rng,
            attacker,
            battle_field=battle_field,
        )
        if tgt is None:
            break
        _run_single_swing(rt, attacker, tgt, attacker_side_idx, defender_side_idx)


def _can_attack(
    minion: BattleMinion,
    side: BattleSide,
    *,
    battle_field: Tuple[BattleSide, BattleSide],
) -> bool:
    return (
        minion.alive
        and attack_value(
            minion,
            side,
            death_resolution=False,
            battle_field=battle_field,
        )
        > 0
    )


def _side_has_attackers(
    side: BattleSide,
    *,
    battle_field: Tuple[BattleSide, BattleSide],
) -> bool:
    return any(_can_attack(m, side, battle_field=battle_field) for m in side.minions)


def _next_attacker(
    side: BattleSide,
    *,
    battle_field: Tuple[BattleSide, BattleSide],
) -> Optional[BattleMinion]:
    n = len(side.minions)
    if n == 0:
        return None
    start = side.cursor % n
    for offset in range(n):
        idx = (start + offset) % n
        if _can_attack(side.minions[idx], side, battle_field=battle_field):
            side.cursor = (idx + 1) % n
            return side.minions[idx]
    return None


def _decide_first_side(
    side0: BattleSide,
    side1: BattleSide,
    p0_has_initiative: bool,
) -> int:
    n0 = side0.alive_count()
    n1 = side1.alive_count()
    if n0 > n1:
        return 0
    if n1 > n0:
        return 1
    return 0 if p0_has_initiative else 1
