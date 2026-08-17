"""What the next tier costs, pinned against the way it is stored.

The price is assembled from four independent pieces — the patch's base cost for
the tier, the discount for every round spent waiting, the one-shot levers
(Deck Swabbie, Chenvaala) and the hero surcharge (Millhouse) — and the way they
combine is easy to get subtly wrong while refactoring where the number lives.

These tests speak only through ``effective_level_up_cost`` /
``accrue_upgrade_discount`` / ``level_up_tavern``, never through the field
behind them, so they hold whether the price is stored and decremented or
derived on demand. That is the point: they are the safety net for changing it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.bg_catalog.patch_context import PatchContext
from src.bg_catalog.ruleset import DEFAULT_RULESET, Ruleset
from src.bg_core.hero import Hero, UpgradeCostSurcharge
from src.bg_lobby.player import PlayerPhase, PlayerState
from src.bg_recruitment import economy

PATCH_74257 = "data/bgcore/19_6_0_74257"


@pytest.fixture(scope="module")
def patch():
    return PatchContext.load(Path(PATCH_74257))


def _player(ruleset: Ruleset = DEFAULT_RULESET, tier: int = 1, **kw):
    base = dict(
        health=40,
        gold=10,
        tavern_tier=tier,
        ruleset=ruleset,
        board=[],
        shop=[None] * 6,
        hand=[None] * 10,
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
    )
    base.update(kw)
    return PlayerState(**base)


def _wait(player: PlayerState, rounds: int) -> None:
    """Spend rounds without upgrading — each one discounts the next tier."""
    for _ in range(rounds):
        economy.accrue_upgrade_discount(player)


def _upgrade(player: PlayerState, patch: PatchContext) -> None:
    economy.level_up_tavern(
        player,
        None,
        rng=np.random.default_rng(0),
        shared_pool=None,
        patch=patch,
    )


# --------------------------------------------------------------------------- #
# The base price is the patch's, and waiting discounts it.
# --------------------------------------------------------------------------- #


def test_price_starts_at_the_patch_base():
    p = _player(tier=1)
    assert economy.effective_level_up_cost(p) == DEFAULT_RULESET.level_up_cost(1)


def test_each_round_waited_takes_one_off():
    p = _player(tier=1)
    start = economy.effective_level_up_cost(p)
    _wait(p, 3)
    assert economy.effective_level_up_cost(p) == start - 3


def test_the_discount_stops_at_free():
    p = _player(tier=1)
    _wait(p, 50)
    assert economy.effective_level_up_cost(p) == 0


def test_waiting_past_free_stays_free():
    """Extra rounds must not bank negative price for later."""
    p = _player(tier=1)
    _wait(p, 50)
    _wait(p, 10)
    assert economy.effective_level_up_cost(p) == 0


def test_a_surcharge_still_applies_after_the_price_bottomed_out():
    """The floor is per-step, not on the total: Millhouse pays 1, not 0.

    Flattening the pieces into one expression collapses this to zero, which is
    the whole reason it is spelled out here.
    """
    hero = Hero("millhouse", "Millhouse Manastorm", passives=(UpgradeCostSurcharge(1),))
    p = _player(tier=1, hero=hero)
    _wait(p, 50)
    assert economy.effective_level_up_cost(p) == 1


# --------------------------------------------------------------------------- #
# Upgrading re-bases the price on the tier reached.
# --------------------------------------------------------------------------- #


def test_upgrading_rebases_the_price_on_the_new_tier(patch):
    ruleset = patch.meta.ruleset
    p = _player(ruleset, tier=1, gold=20)
    _wait(p, 2)
    _upgrade(p, patch)
    assert p.tavern_tier == 2
    assert economy.effective_level_up_cost(p) == ruleset.level_up_cost(2)


def test_the_waiting_discount_does_not_carry_across_an_upgrade(patch):
    """Rounds waited at tier 1 must not pre-discount tier 2."""
    ruleset = patch.meta.ruleset
    early = _player(ruleset, tier=1, gold=20)
    _upgrade(early, patch)

    patient = _player(ruleset, tier=1, gold=20)
    _wait(patient, 4)
    _upgrade(patient, patch)

    assert economy.effective_level_up_cost(patient) == economy.effective_level_up_cost(early)


def test_the_price_comes_from_the_package_not_the_default(patch):
    """19.6.0 charges 9 for tier 5 where the default ruleset charges 11."""
    ruleset = patch.meta.ruleset
    assert ruleset.level_up_cost(4) == 9
    assert DEFAULT_RULESET.level_up_cost(4) == 11

    p = _player(ruleset, tier=3, gold=30)
    _upgrade(p, patch)  # now tier 4, priced for the step to 5
    assert economy.effective_level_up_cost(p) == 9


def test_the_top_tier_has_no_price_to_discount(patch):
    """At the ceiling there is no next tier; waiting cannot move the number."""
    ruleset = patch.meta.ruleset
    p = _player(ruleset, tier=ruleset.max_tier)
    before = economy.effective_level_up_cost(p)
    _wait(p, 5)
    assert economy.effective_level_up_cost(p) == before


# --------------------------------------------------------------------------- #
# One-shot levers ride on top and are consumed by the upgrade.
# --------------------------------------------------------------------------- #


def test_a_one_shot_discount_comes_off_the_current_price(patch):
    ruleset = patch.meta.ruleset
    p = _player(ruleset, tier=1, gold=20, upgrade_cost_delta=-2)
    assert economy.effective_level_up_cost(p) == ruleset.level_up_cost(1) - 2


def test_a_one_shot_discount_is_spent_by_the_upgrade(patch):
    ruleset = patch.meta.ruleset
    p = _player(ruleset, tier=1, gold=20, upgrade_cost_delta=-2)
    _upgrade(p, patch)
    assert p.upgrade_cost_delta == 0
    assert economy.effective_level_up_cost(p) == ruleset.level_up_cost(2)


def test_an_accumulated_hero_discount_is_spent_by_the_upgrade(patch):
    """Chenvaala banks a discount across rounds; the upgrade consumes all of it."""
    ruleset = patch.meta.ruleset
    p = _player(ruleset, tier=1, gold=20, hero_upgrade_discount=3, hero=Hero("x", "X"))
    assert economy.effective_level_up_cost(p) == ruleset.level_up_cost(1) - 3
    _upgrade(p, patch)
    assert p.hero_upgrade_discount == 0
    assert economy.effective_level_up_cost(p) == ruleset.level_up_cost(2)


def test_levers_never_make_the_price_negative(patch):
    ruleset = patch.meta.ruleset
    p = _player(ruleset, tier=1, upgrade_cost_delta=-99)
    assert economy.effective_level_up_cost(p) == 0


def test_the_ceiling_costs_nothing_because_there_is_nothing_to_buy(patch):
    """Both observation blocks must agree there is no next tier to price.

    They did not: the base observation zeroes the cost at the ceiling
    (``obs.py``), while the hero block asks ``effective_level_up_cost``
    unguarded (``obs_v5_heroes.py``). While the price was a stored field, the
    ceiling left it holding the price of the upgrade already bought — frozen,
    because the upgrade skipped the write at max tier — so the hero block
    reported a seat could buy a seventh tier for 9 gold. Deriving the price
    settled it at 0 and the two blocks now agree, which the golden trace caught
    as three diverged lobbies.
    """
    ruleset = patch.meta.ruleset
    p = _player(ruleset, tier=ruleset.max_tier, gold=50)
    assert ruleset.level_up_cost(ruleset.max_tier) == 0, "no price is declared past the top"
    assert economy.effective_level_up_cost(p) == 0
