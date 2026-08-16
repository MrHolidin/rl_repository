"""Dead minions leave the board; the board closes up behind them.

Corpses used to stay in ``BattleSide.minions`` forever, with ``alive`` as a
derived property, so every one of the ~35 loops over a side had to remember to
filter — and a corpse kept occupying a slot, which meant two survivors with a
dead minion between them were never adjacent. That is not how the game works:
the board compacts on death.

The body is still needed after it dies (its deathrattle summons into the slot
it vacated, Reborn returns there, and queued events resolve it by
``instance_id``), so it moves to ``graveyard`` carrying ``death_pos`` rather
than being dropped.

Shape borrowed from twanvl/hearthstone-battlegrounds-simulator, which sweeps
the dead out in board order, records each one's position in the *cleaned*
board, and only then runs the death triggers.
"""

from __future__ import annotations

import numpy as np
import pytest

import src.envs.minibg  # noqa: F401  (breaks a circular import at collection)
from src.bg_catalog.cards import make_minion
from src.bg_catalog.patch_context import load_patch_context
from src.bg_combat.battle.auras import attack_value
from src.bg_combat.battle.sides import _build_side
from src.bg_combat.battle.state import BattleSide, _CombatRuntime

PATCH_15_6 = "data/bgcore/15_6_2_36393"
DIRE_WOLF = "EX1_162"  # Adjacent minions have +1 Attack
TIDEHUNTER = "EX1_506"


@pytest.fixture(scope="module")
def ctx():
    return load_patch_context(PATCH_15_6)


def _runtime(ctx, board):
    rt = _CombatRuntime(
        sides=(BattleSide([]), BattleSide([])),
        rng=np.random.default_rng(0),
        combat_board_max=7,
        damage_cap=15,
        patch=ctx,
    )
    rt.sides = (_build_side(board, rt), _build_side([], rt))
    return rt


def _body(ctx, attack=2, health=9):
    m = make_minion(TIDEHUNTER, patch=ctx)
    m.base_attack, m.base_health = attack, health
    return m


def test_reap_removes_the_body_and_records_its_slot(ctx):
    rt = _runtime(ctx, [_body(ctx), _body(ctx), _body(ctx)])
    side = rt.side(0)
    victim = side.minions[1]

    victim.current_health = 0
    taken = side.reap_dead()

    assert taken == [victim]
    assert victim not in side.minions
    assert side.graveyard == [victim]
    assert victim.death_pos == 1
    assert len(side.minions) == 2


def test_board_closes_up_so_neighbours_become_adjacent(ctx):
    """[X][Y][DireWolf]: X is not adjacent to the wolf until Y dies."""
    rt = _runtime(ctx, [_body(ctx), _body(ctx), make_minion(DIRE_WOLF, patch=ctx)])
    side = rt.side(0)
    x, y, _wolf = side.minions

    def x_attack():
        return attack_value(x, side, death_resolution=False, battle_field=rt.sides)

    assert x_attack() == 2, "X starts two slots from the wolf"
    y.current_health = 0
    side.reap_dead()
    assert x_attack() == 3, "with the corpse gone X is adjacent and gets +1"


def test_cursor_follows_the_removals(ctx):
    """``cursor`` indexes ``minions``, so it has to move with them."""
    rt = _runtime(ctx, [_body(ctx) for _ in range(4)])
    side = rt.side(0)

    # A body left of the pointer pulls it left.
    side.cursor = 3
    side.minions[0].current_health = 0
    side.reap_dead()
    assert side.cursor == 2

    # A body *at* the pointer leaves it on whoever slid into that slot.
    rt2 = _runtime(ctx, [_body(ctx) for _ in range(4)])
    side2 = rt2.side(0)
    side2.cursor = 1
    survivor_after = side2.minions[2]
    side2.minions[1].current_health = 0
    side2.reap_dead()
    assert side2.minions[side2.cursor] is survivor_after


def test_cursor_stays_in_range_when_the_board_empties(ctx):
    rt = _runtime(ctx, [_body(ctx), _body(ctx)])
    side = rt.side(0)
    side.cursor = 1
    for m in side.minions:
        m.current_health = 0
    side.reap_dead()
    assert side.minions == []
    assert side.cursor == 0


def test_find_minion_still_resolves_a_body_in_the_graveyard(ctx):
    """Queued events outlive the body and look it up by instance_id."""
    rt = _runtime(ctx, [_body(ctx), _body(ctx)])
    side = rt.side(0)
    victim = side.minions[0]
    vid = victim.instance_id

    victim.current_health = 0
    side.reap_dead()

    assert rt.find_minion(0, vid) is victim


def test_deathrattle_token_takes_the_vacated_slot(ctx):
    """Harvest Golem's token appears where the golem stood, not at the end."""
    from src.bg_combat.battle.effects import _deal_damage_to_battle_minion
    from src.bg_combat.battle.engine import _dispatch

    ctx74 = load_patch_context("data/bgcore/19_6_0_74257")
    board = [_body(ctx74), make_minion("EX1_556", patch=ctx74), _body(ctx74)]
    rt = _runtime(ctx74, board)
    side = rt.side(0)
    golem = side.minions[1]

    _deal_damage_to_battle_minion(rt, 0, golem, 99)
    while rt.queue:
        _dispatch(rt, rt.queue.popleft())

    assert golem not in side.minions
    assert golem.death_pos == 1
    assert side.minions[1].template.name == "Damaged Golem"


def test_token_in_a_vacated_slot_attacks_this_pass(ctx):
    """A deathrattle token inherits the dead minion's place in the rotation.

    The body that died was the one the attack pointer was on, so nobody is
    waiting on that slot: the token that fills it swings this pass rather than
    waiting for the wrap-around.
    """
    from src.bg_combat.battle.effects import _deal_damage_to_battle_minion
    from src.bg_combat.battle.engine import _dispatch, _next_attacker

    ctx74 = load_patch_context("data/bgcore/19_6_0_74257")
    # Harvest Golem leads; it dies and leaves a Damaged Golem in its slot.
    board = [make_minion("EX1_556", patch=ctx74), _body(ctx74), _body(ctx74)]
    rt = _runtime(ctx74, board)
    side = rt.side(0)
    golem = side.minions[0]

    first = _next_attacker(side, battle_field=rt.sides)
    assert first is golem, "the golem leads the rotation"
    assert side.cursor == 1

    _deal_damage_to_battle_minion(rt, 0, golem, 99)
    while rt.queue:
        _dispatch(rt, rt.queue.popleft())

    assert golem not in side.minions
    token = side.minions[0]
    assert token.template.name == "Damaged Golem"
    assert side.cursor == 0, "pointer stayed on the vacated slot"
    assert _next_attacker(side, battle_field=rt.sides) is token


def test_token_between_living_minions_waits_its_turn(ctx):
    """The other half of the rule: nobody jumps a living minion's turn."""
    from src.bg_combat.battle.engine import _next_attacker
    from src.bg_combat.battle.summon import _summon_insert

    rt = _runtime(ctx, [_body(ctx), _body(ctx), _body(ctx)])
    side = rt.side(0)
    first = _next_attacker(side, battle_field=rt.sides)
    assert first is side.minions[0] and side.cursor == 1
    waiting = side.minions[1]

    # Not a death: a plain summon dropped where the pointer is.
    assert not rt.in_death_resolution
    _summon_insert(rt, 0, make_minion(TIDEHUNTER, patch=ctx), 1)

    assert side.cursor == 2, "pointer shifted so the waiting minion keeps its turn"
    assert _next_attacker(side, battle_field=rt.sides) is waiting


def test_overkill_excess_lands_after_the_body_left_the_board(ctx):
    """Overkill resolves once the victim is already in the graveyard.

    The excess used to be located with ``side.minions.index(victim)``, which
    raises once the body is off the board -- and the handler returned on that,
    throwing the damage away silently. Wildfire Elemental became inert without
    a single test noticing; the boards simply had more survivors.
    """
    import numpy as np

    from src.bg_combat.battle.effects import _deal_damage_to_battle_minion
    from src.bg_combat.battle.engine import _dispatch
    from src.bg_combat.battle.events import Overkill

    ctx74 = load_patch_context("data/bgcore/19_6_0_74257")
    left = make_minion(TIDEHUNTER, patch=ctx74)
    left.base_attack, left.base_health = 0, 20
    right = make_minion(TIDEHUNTER, patch=ctx74)
    right.base_attack, right.base_health = 0, 20
    victim = make_minion("BGS_014", patch=ctx74)  # Imprisoner: deathrattle summons an Imp
    wildfire = make_minion("BGS_126", patch=ctx74)  # Wildfire Elemental

    rt = _runtime(ctx74, [left, victim, right])
    rt.sides = (rt.side(0), _build_side([wildfire], rt))
    side = rt.side(0)
    body = side.minions[1]

    rt.queue.append(Overkill(0, body.instance_id, 1, rt.side(1).minions[0].instance_id, 8))
    _deal_damage_to_battle_minion(rt, 0, body, 9)
    while rt.queue:
        _dispatch(rt, rt.queue.popleft())

    names = [m.template.name for m in side.minions]
    assert "Imp" in names, "the deathrattle still summoned into the vacated slot"
    neighbours = [m for m in side.minions if m.template.name == "Murloc Tidehunter"]
    assert len(neighbours) == 2
    damaged = [m for m in neighbours if m.current_health < 20]
    assert len(damaged) == 1, "Wildfire hits one random neighbour, and it must hit one"
    assert damaged[0].current_health == 12, "the full excess landed"

    imp = next(m for m in side.minions if m.template.name == "Imp")
    assert imp.current_health == imp.template.max_health, (
        "the token that filled the slot is summoned after Overkill resolves and "
        "must not soak the excess"
    )
