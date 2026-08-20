"""Random battles, and the two bugs a first run of them found.

Every dispatcher in the engine is deliberately loud, so a fuzz run is a real
coverage test: an ability nobody handles takes the fight down the first time it
is drawn. The script is ``scripts/fuzz_bglike_battles.py``; this keeps a short
run in the suite and pins what it caught.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.bg_catalog.patch_context import PatchContext
from src.envs.minibg.summon_pool import summon_pool_for

PATCH_DIRS = (
    "data/bgcore/36_2_0_248348",
    "data/bgcore/19_6_0_74257",
    "data/bgcore/15_6_2_36393",
)


@pytest.fixture(scope="module")
def patch() -> PatchContext:
    return PatchContext.load(Path(PATCH_DIRS[0]))


@pytest.mark.parametrize("patch_dir", PATCH_DIRS)
def test_the_summon_pool_only_offers_cards_that_can_be_made(patch_dir):
    """A Duos card has no template, so summoning one is summoning nothing.

    ``PatchContext`` filters them when it builds templates and the summon pool
    did not, so Ghastcoiler could roll one and raise mid-fight.
    """
    ctx = PatchContext.load(Path(patch_dir))
    pool = summon_pool_for(None, False, False, None, None, patch=ctx)
    assert pool
    assert [cid for cid in pool if cid not in ctx.templates] == []


def test_a_rally_that_raises_the_tribe_gift_has_a_combat_handler(patch):
    """Moat Custodian's Rally writes to the seat, and combat had no branch."""
    from src.bg_combat.battle.seat import RecordingSeat
    from tests.minibg_helpers import simulate_battle
    from src.bg_core.minion import Minion

    custodian = patch.make_minion("BG36_351")
    custodian.bonus_health += 40
    seat = RecordingSeat()
    simulate_battle(
        [custodian],
        [Minion(card_id="wall", base_attack=0, base_health=30, tier=1)],
        p0_has_initiative=True,
        rng=np.random.default_rng(0),
        patch=patch,
        seats=(seat, RecordingSeat()),
    )
    assert seat.tribe_gifts  # it reached the seat rather than raising


@pytest.mark.parametrize("patch_dir", PATCH_DIRS)
def test_random_battles_do_not_crash(patch_dir):
    """A short run of the fuzz script, so a new binding cannot ship a card that
    raises the first time it is played."""
    from scripts.fuzz_bglike_battles import _one_battle

    ctx = PatchContext.load(Path(patch_dir))
    for seed in range(60):
        _one_battle(ctx, seed)
