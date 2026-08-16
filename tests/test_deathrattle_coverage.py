"""Every shipped ON_DEATH ability must have a deathrattle handler.

King Bagurgle shipped for the entire life of the 19.6 package with half its
text inert: ``_fire_deathrattle`` resolved handlers with
``_DEATHRATTLE_HANDLERS.get(type(effect))`` and skipped a miss without a word,
and that effect type was never registered. Nothing failed, no test went red,
the card just quietly did less than it said.

The dispatch now raises on a miss, and this pins the shipped catalogs so the
raise can never actually fire in a game or a training run — a new binding that
forgets to register its effect fails here first.
"""

from __future__ import annotations

import pytest

from src.bg_catalog.patch_context import load_patch_context
from src.bg_combat.battle.effects import _DEATHRATTLE_HANDLERS
from src.bg_core.effects import Trigger

PATCH_DIRS = ("data/bgcore/19_6_0_74257", "data/bgcore/15_6_2_36393")


@pytest.mark.parametrize("patch_dir", PATCH_DIRS)
def test_every_on_death_ability_has_a_handler(patch_dir):
    ctx = load_patch_context(patch_dir)
    unhandled = [
        (cid, ctx.describe(cid).name, type(ab.effect).__name__)
        for cid, tpl in sorted(ctx.templates.items())
        for ab in tpl.abilities
        if ab.trigger is Trigger.ON_DEATH
        and type(ab.effect) not in _DEATHRATTLE_HANDLERS
    ]
    assert not unhandled, (
        "ON_DEATH abilities with no handler — these cards silently do nothing:\n"
        + "\n".join(f"  {c} {n}: {e}" for c, n, e in unhandled)
    )


@pytest.mark.parametrize("patch_dir", PATCH_DIRS)
def test_golden_forms_are_covered_too(patch_dir):
    """Triple-forged goldens rebuild their abilities, so check those as well."""
    ctx = load_patch_context(patch_dir)
    unhandled = []
    for cid in sorted(ctx.templates):
        try:
            abilities = ctx.triple_merge_golden_abilities(cid)
        except Exception:
            continue
        for ab in abilities:
            if ab.trigger is Trigger.ON_DEATH and type(ab.effect) not in _DEATHRATTLE_HANDLERS:
                unhandled.append((cid, type(ab.effect).__name__))
    assert not unhandled, f"golden ON_DEATH with no handler: {unhandled}"


def test_dispatch_raises_instead_of_skipping():
    """The behaviour that makes the above enforceable rather than advisory."""
    import numpy as np

    from src.bg_catalog.cards import make_minion
    from src.bg_combat.battle import simulate_battle
    from src.bg_core.effects import Ability, HeroImmuneAura

    ctx = load_patch_context("data/bgcore/19_6_0_74257")
    # HeroImmuneAura is a real effect that is not (and should not be) a
    # deathrattle handler — standing in for "a binding forgot to register".
    assert HeroImmuneAura not in _DEATHRATTLE_HANDLERS
    doomed = make_minion("EX1_506", patch=ctx)
    doomed.abilities = (Ability(Trigger.ON_DEATH, HeroImmuneAura()),)
    killer = make_minion("EX1_506", patch=ctx)
    killer.bonus_attack += 20

    with pytest.raises(KeyError, match="no deathrattle handler for HeroImmuneAura"):
        simulate_battle(
            [doomed],
            [killer],
            p0_has_initiative=False,
            rng=np.random.default_rng(0),
            combat_board_max=7,
            damage_cap=15,
            max_board_slots=7,
            patch=ctx,
        )
