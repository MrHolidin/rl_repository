"""Deathrattle / self-damaged summons spawn at the source's board slot.

Real-BG rule: tokens appear where the minion died (or right of the damaged
source), not at the right end of the board. Dead bodies stay in the runtime
minion list, so "in place" means: directly after the dead body, in summon
order, with everything to the right shifted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import src.envs  # noqa: F401  (break circular import)
from src.bg_catalog.cards import make_minion
from src.bg_catalog.patch_context import PatchContext
from src.bg_combat.battle import (
    BattleSide,
    _build_side,
    _CombatRuntime,
    _deal_damage_to_battle_minion,
    _dispatch,
    _next_attacker,
    _run_single_swing,
    _summon_insert,
)

PATCH_CTX = PatchContext.load(Path("data/bgcore/19_6_0_74257"))

_BY_NAME = {}
for _cid, _d in sorted(PATCH_CTX.descriptions.items()):
    if _cid in PATCH_CTX.pool_ids and _d.name not in _BY_NAME:
        _BY_NAME[_d.name] = _cid


def _mk(name: str):
    return make_minion(_BY_NAME[name], patch=PATCH_CTX)


def _runtime(board0, board1):
    rt = _CombatRuntime(
        sides=(BattleSide(), BattleSide()),
        rng=np.random.default_rng(0),
        combat_board_max=7,
        damage_cap=15,
        patch=PATCH_CTX,
    )
    rt.sides = (_build_side(board0, rt), _build_side(board1, rt))
    return rt


def _drain(rt):
    while rt.queue:
        _dispatch(rt, rt.queue.popleft())


def _alive_names(side):
    return [m.template.name for m in side.minions if m.alive]


def test_deathrattle_token_spawns_in_dead_minions_slot():
    rt = _runtime(
        [_mk("Dragonspawn Lieutenant"), _mk("Harvest Golem"), _mk("Vulgar Homunculus")],
        [_mk("Murloc Tidehunter")],
    )
    side = rt.side(0)
    golem = side.minions[1]
    _deal_damage_to_battle_minion(rt, 0, golem, 99)
    _drain(rt)
    assert _alive_names(side) == [
        "Dragonspawn Lieutenant",
        "Damaged Golem",
        "Vulgar Homunculus",
    ]
    # dead body stays in the list, token directly after it
    assert side.minions[1] is golem and not golem.alive
    assert side.minions[2].template.name == "Damaged Golem"


def test_multi_token_deathrattle_spawns_in_order_in_place():
    rt = _runtime(
        [_mk("Dragonspawn Lieutenant"), _mk("Replicating Menace"), _mk("Vulgar Homunculus")],
        [_mk("Murloc Tidehunter")],
    )
    side = rt.side(0)
    menace = side.minions[1]
    _deal_damage_to_battle_minion(rt, 0, menace, 99)
    _drain(rt)
    names = _alive_names(side)
    assert names[0] == "Dragonspawn Lieutenant"
    assert names[-1] == "Vulgar Homunculus"
    assert names[1:-1] == ["Microbot"] * 3


def test_khadgar_doubled_tokens_stay_in_place():
    rt = _runtime(
        [_mk("Khadgar"), _mk("Harvest Golem"), _mk("Vulgar Homunculus")],
        [_mk("Murloc Tidehunter")],
    )
    side = rt.side(0)
    golem = side.minions[1]
    _deal_damage_to_battle_minion(rt, 0, golem, 99)
    _drain(rt)
    assert _alive_names(side) == [
        "Khadgar",
        "Damaged Golem",
        "Damaged Golem",
        "Vulgar Homunculus",
    ]


def test_self_damaged_summon_spawns_right_of_source():
    rt = _runtime(
        [_mk("Imp Gang Boss"), _mk("Vulgar Homunculus")],
        [_mk("Murloc Tidehunter")],
    )
    side = rt.side(0)
    boss = side.minions[0]
    attacker = rt.side(1).minions[0]
    # ON_SELF_DAMAGED fires on strike damage: enemy 2/1 hits the 2/4 boss.
    _run_single_swing(rt, attacker, boss, 1, 0)
    assert _alive_names(side) == ["Imp Gang Boss", "Imp", "Vulgar Homunculus"]


def test_insert_before_cursor_shifts_cursor_no_skip_no_repeat():
    rt = _runtime(
        [_mk("Murloc Tidehunter"), _mk("Vulgar Homunculus"), _mk("Dragonspawn Lieutenant")],
        [_mk("Murloc Tidehunter")],
    )
    side = rt.side(0)
    sides = (rt.side(0), rt.side(1))
    first = _next_attacker(side, battle_field=sides)
    assert first is side.minions[0]
    assert side.cursor == 1
    # token inserted at index 1 (before/at cursor) must not become the next
    # attacker out of order: cursor shifts so the pre-insert minion at index 1
    # attacks next.
    homunculus = side.minions[1]
    tok = make_minion(_BY_NAME["Murloc Tidehunter"], patch=PATCH_CTX)
    _summon_insert(rt, 0, tok, 1)
    assert side.cursor == 2
    second = _next_attacker(side, battle_field=sides)
    assert second is homunculus
    # the inserted token attacks on the wrap-around pass
    third = _next_attacker(side, battle_field=sides)
    assert third is side.minions[3]  # Dragonspawn Lieutenant (shifted right)
    fourth = _next_attacker(side, battle_field=sides)
    assert fourth.template.card_id == tok.card_id
