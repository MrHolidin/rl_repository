"""The active patch's ruleset, not the default one, drives tavern upgrades.

Rules the engine reads from module constants are rules a patch package cannot
change. 74257 prices the tier 4→5 upgrade at 9 in its own meta.json while the
module table says 11 — the engine charged 11, and nothing noticed, because
nothing asserted that a package's numbers actually reach the shop.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.envs.bglike.game import BGLikeGame  # imported first: primes env packages
from src.bg_catalog.patch_context import load_patch_context
from src.bg_player_turn.engine import PlayerTurnEngine
from src.bg_recruitment import economy
from src.envs.bglike.actions import Action

_PATCH_74257 = "data/bgcore/19_6_0_74257"
_PATCH_36393 = "data/bgcore/15_6_2_36393"


def _seat_at_tier(patch_dir: str, tier: int):
    """A seat levelled up to ``tier`` with gold to spare, plus its patch."""
    ctx = load_patch_context(patch_dir)
    game = BGLikeGame(seed=0, patch_dir=patch_dir)
    state = game.initial_state()
    player = state.players[0]
    rng = np.random.default_rng(0)
    while player.tavern_tier < tier:
        player.gold = 50
        economy.level_up_tavern(
            player, None, rng=rng, shared_pool=state.shared_pool, patch=ctx
        )
    return player, ctx, state


@pytest.mark.parametrize("patch_dir", [_PATCH_74257, _PATCH_36393])
def test_next_upgrade_price_comes_from_the_package(patch_dir):
    ctx = load_patch_context(patch_dir)
    costs = ctx.meta.ruleset.level_up_costs
    for tier in range(2, int(ctx.meta.ruleset.max_tier)):
        player, _ctx, _state = _seat_at_tier(patch_dir, tier)
        assert player.next_tier_up_cost == costs[tier], (
            f"{patch_dir}: at tier {tier} the next upgrade costs "
            f"{player.next_tier_up_cost}, package says {costs[tier]}"
        )


def test_74257_tier_five_upgrade_costs_nine():
    """The concrete regression: the module table would charge 11 here."""
    player, ctx, _state = _seat_at_tier(_PATCH_74257, 4)
    assert ctx.meta.ruleset.level_up_costs[4] == 9
    assert player.next_tier_up_cost == 9
    assert economy.effective_level_up_cost(player) == 9


def test_upgrade_is_illegal_at_the_packages_tier_ceiling():
    ctx = load_patch_context(_PATCH_74257)
    player, _ctx, _state = _seat_at_tier(_PATCH_74257, ctx.meta.ruleset.max_tier)
    player.gold = 50
    legal = PlayerTurnEngine().legal_actions(player, ctx.meta.ruleset)
    assert int(Action.LEVEL_UP) not in legal


def test_a_seventh_tier_is_reachable_when_the_ruleset_declares_one():
    """Tier 7 needs no engine change — only a package that prices and allows it."""
    ctx = load_patch_context(_PATCH_74257)
    tier7 = replace(
        ctx.meta.ruleset,
        max_tier=7,
        level_up_costs={**dict(ctx.meta.ruleset.level_up_costs), 6: 11},
        # A tier-7 tavern shows a seventh offer; the layout has to have the slot.
    )
    ctx7 = replace(ctx, meta=replace(ctx.meta, ruleset=tier7, layout=replace(
        ctx.meta.layout,
        num_tiers=7,
        max_shop_slots=7,
        shop_offers_by_tier={**dict(ctx.meta.layout.shop_offers_by_tier), 7: 7},
    )))

    game = BGLikeGame(seed=0, patch_dir=_PATCH_74257)
    state = game.initial_state()
    player = state.players[0]
    # A seat plays under the rules it was dealt: the price of its next tier is
    # derived from this, so a synthetic package has to reach the seat too.
    player.ruleset = tier7
    rng = np.random.default_rng(0)
    while player.tavern_tier < 6:
        player.gold = 50
        economy.level_up_tavern(
            player, None, rng=rng, shared_pool=state.shared_pool, patch=ctx7
        )

    player.gold = 50
    assert int(Action.LEVEL_UP) in PlayerTurnEngine().legal_actions(player, tier7)
    assert player.next_tier_up_cost == 11

    economy.level_up_tavern(
        player, None, rng=rng, shared_pool=state.shared_pool, patch=ctx7
    )
    assert player.tavern_tier == 7
    # And seven is the ceiling now.
    player.gold = 50
    assert int(Action.LEVEL_UP) not in PlayerTurnEngine().legal_actions(player, tier7)


def test_upgrade_is_legal_one_tier_below_the_ceiling():
    ctx = load_patch_context(_PATCH_74257)
    player, _ctx, _state = _seat_at_tier(_PATCH_74257, ctx.meta.ruleset.max_tier - 1)
    player.gold = 50
    legal = PlayerTurnEngine().legal_actions(player, ctx.meta.ruleset)
    assert int(Action.LEVEL_UP) in legal
