"""Every shipped effect on a shop trigger must be handled or declared inert.

The tavern half of the engine had the same hole the deathrattle table had
before King Bagurgle was found: ``apply_shop_effect`` was a chain of
``elif isinstance`` with nothing at the end, so an effect nobody had written a
branch for fell straight through. That is indistinguishable from an effect
that is *meant* to do nothing here -- and twelve of them are, because a
targeted battlecry is applied off the placement action and a Pogo-Hopper is
applied by ``fire_on_place`` itself before it delegates.

Now the chain ends in a raise and the deliberate cases are named in
``_HANDLED_ELSEWHERE``. This pins the shipped catalogs so the raise cannot
fire in a game or a training run: a card whose effect nobody routed fails
here first, and so does an entry in ``_HANDLED_ELSEWHERE`` that stops being
true.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from src.bg_catalog.patch_context import load_patch_context
from src.bg_core.effects import Trigger
from src.bg_recruitment.shop_triggers import _HANDLED_ELSEWHERE

PATCH_DIRS = ("data/bgcore/19_6_0_74257", "data/bgcore/15_6_2_36393")

#: Triggers whose abilities reach ``apply_shop_effect``. ON_TURN_START and
#: ON_TURN_END are absent on purpose: their loops dispatch a fixed handful of
#: effect types themselves and never delegate the rest.
ROUTED_TO_DISPATCH = (
    Trigger.ON_SELL,
    Trigger.ON_FRIENDLY_BOUGHT,
    Trigger.ON_PLACE,
    Trigger.AFTER_FRIENDLY_MINION_PLACED,
)


def _branch_types() -> set[str]:
    """Effect names ``apply_shop_effect`` tests for, read off the source.

    Reading the chain rather than keeping a second list is the point: a branch
    that is added or deleted cannot drift away from this test.
    """
    src = pathlib.Path("src/bg_recruitment/shop_triggers.py").read_text()
    fn = next(
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "apply_shop_effect"
    )
    return {
        n.args[1].id
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", "") == "isinstance"
        and isinstance(n.args[1], ast.Name)
    }


@pytest.mark.parametrize("patch_dir", PATCH_DIRS)
def test_every_shop_effect_is_handled_or_declared_inert(patch_dir):
    ctx = load_patch_context(patch_dir)
    known = _branch_types() | {t.__name__ for t in _HANDLED_ELSEWHERE}
    unhandled = sorted(
        {
            (type(ab.effect).__name__, ctx.describe(cid).name)
            for cid, tpl in ctx.templates.items()
            for ab in tpl.abilities
            if ab.trigger in ROUTED_TO_DISPATCH
            and type(ab.effect).__name__ not in known
        }
    )
    assert not unhandled, (
        "effects reaching the shop dispatcher with no branch and no "
        "_HANDLED_ELSEWHERE entry:\n"
        + "\n".join(f"  {e} ({c})" for e, c in unhandled)
    )


#: Entries that no *shipped* package reaches yet because the engine learned the
#: mechanic before a package that prints it exists. Distinct from a stale entry,
#: which used to be reachable and no longer is — that is what the test below is
#: for. A name leaves this set when a package starts using it.
AHEAD_OF_THE_PACKAGES = {"ChooseOneEffect"}


def test_handled_elsewhere_entries_are_all_still_reachable():
    """An entry that no card can produce any more is a stale excuse."""
    reachable = set(AHEAD_OF_THE_PACKAGES)
    for patch_dir in PATCH_DIRS:
        ctx = load_patch_context(patch_dir)
        for tpl in ctx.templates.values():
            for ab in tpl.abilities:
                if ab.trigger in ROUTED_TO_DISPATCH:
                    reachable.add(type(ab.effect).__name__)
    stale = sorted(t.__name__ for t in _HANDLED_ELSEWHERE if t.__name__ not in reachable)
    assert not stale, f"_HANDLED_ELSEWHERE entries no shipped card reaches: {stale}"


def test_dispatch_raises_instead_of_falling_through():
    """The behaviour that makes the checks above enforceable rather than advisory."""
    import numpy as np

    from src.bg_lobby.player import PlayerPhase, PlayerState
    from src.bg_recruitment.shop_triggers import ShopTriggers, UnhandledShopEffect

    class NotAnEffect:
        pass

    ctx = load_patch_context(PATCH_DIRS[0])
    engine = ShopTriggers(rng=np.random.default_rng(0), patch=ctx)
    source = ctx.make_minion("EX1_506")
    player = PlayerState(
        health=40,
        hero_damage_taken_total=0,
        gold=0,
        tavern_tier=1,
        board=[source],
        shop=[None] * 6,
        hand=[None] * 10,
        phase=PlayerPhase.DONE,
        shop_actions_used=0,
    )
    with pytest.raises(UnhandledShopEffect, match="NotAnEffect"):
        engine.apply_shop_effect(player, source, NotAnEffect(), None)
