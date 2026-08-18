"""Every PlayerState field must survive a copy, and adding one must force a choice.

The copy runs once per *action*, not per turn, so a dropped field is not a
rounding error: it means the value exists for one decision after it is written
and is gone for the rest of the shop turn. The hand-written copy this guards
listed 22 of 38 fields.

The field walk is the point. A test that names fields would have to be updated
alongside the copy — the same coupling that broke — so it enumerates
``dataclasses.fields`` instead and fails on anything new that is neither carried
nor explicitly listed here.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.bg_catalog.patch_context import load_patch_context
from src.bg_core.minion import Race
from src.bg_lobby.player import (
    PendingChoice,
    PendingChoiceKind,
    PlayerPhase,
    PlayerState,
    copy_player_state,
)

PATCH_DIR = "data/bgcore/19_6_0_74257"

# Fields the copy rebuilds instead of carrying: both point AT board minions and
# must be re-aimed at the clones, so identity differs by design.
_REBUILT = {"placed_minion_board_index", "placed_minion_pending_after"}


@pytest.fixture(scope="module")
def patch():
    return load_patch_context(PATCH_DIR)


def _distinct_value(f: dataclasses.Field, patch):
    """A value that differs from the field's default, per declared type."""
    name, t = f.name, str(f.type)
    if name == "board":
        return [patch.make_minion("EX1_103")]
    if name in ("shop", "hand"):
        return [patch.make_minion("EX1_103"), None]
    if name == "last_round_tribe_counts":
        return {Race.MURLOC: 4}
    if name == "bought_tribe_counts":
        return {Race.MURLOC: 2, None: 1}
    if name == "game_counts":
        return {"summoned:BG_TTN_401": 3}
    if name == "standing_bonuses":
        from src.bg_core.effects import ScopeKind
        from src.bg_recruitment.standing_bonuses import BonusScope

        return {BonusScope(ScopeKind.TRIBE, Race.UNDEAD): (1, 0)}
    if name == "tribe_pref":
        return (0.25, -0.5, 0.75, -1.0, 0.0, 0.125, -0.875)
    if name == "last_opponent_board":
        return (patch.make_minion("EX1_103"),)
    if name == "last_battle_snapshots":
        return ()
    if name == "phase":
        return PlayerPhase.DONE
    if name == "hero":
        return None
    if name == "pending_choice":
        return PendingChoice(PendingChoiceKind.DISCOVER_MURLOC, ("a", "b", "c"), 0, (), None)
    if name == "shop_frozen":
        return (True, False, False, False, False, False)
    if name == "battle_history":
        return (0.5, -0.25)
    if "bool" in t:
        return True
    if "float" in t:
        return 3.5
    if "int" in t:
        return 7
    if "str" in t:
        return "x"
    return None


def test_every_field_survives_a_copy(patch):
    base = PlayerState(
        health=40, gold=3, tavern_tier=1,         board=[], shop=[], hand=[], phase=PlayerPhase.SHOP, shop_actions_used=0,
    )
    fields = dataclasses.fields(PlayerState)
    for f in fields:
        v = _distinct_value(f, patch)
        if v is not None:
            setattr(base, f.name, v)

    got = copy_player_state(base)
    dropped = []
    for f in fields:
        if f.name in _REBUILT:
            continue
        want, have = getattr(base, f.name), getattr(got, f.name)
        if f.name in ("board", "shop", "hand", "last_opponent_board"):
            same = [None if m is None else m.card_id for m in want] == [
                None if m is None else m.card_id for m in have
            ]
        else:
            same = want == have
        if not same:
            dropped.append(f.name)
    assert not dropped, f"copy_player_state dropped: {dropped}"


def test_copy_is_isolated_from_the_original(patch):
    base = PlayerState(
        health=40, gold=3, tavern_tier=1,         board=[patch.make_minion("EX1_103")], shop=[], hand=[],
        phase=PlayerPhase.SHOP, shop_actions_used=0,
        last_round_tribe_counts={Race.MURLOC: 1},
    )
    got = copy_player_state(base)

    got.gold = 999
    got.board[0].bonus_attack += 7
    got.board.append(patch.make_minion("EX1_103"))
    got.last_round_tribe_counts[Race.BEAST] = 3

    assert base.gold == 3
    assert base.board[0].bonus_attack == 0
    assert len(base.board) == 1
    assert Race.BEAST not in base.last_round_tribe_counts


def test_a_new_field_cannot_be_added_without_a_decision():
    """Guards the guard: the walk must actually cover the whole dataclass."""
    walked = {f.name for f in dataclasses.fields(PlayerState)}
    assert len(walked) >= 38, "PlayerState shrank — check the copy still covers it"
    assert _REBUILT <= walked
