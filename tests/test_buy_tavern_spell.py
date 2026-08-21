"""Buying the Tavern spell off the counter.

The spell sits beside the minion row rather than in it, so buying it is its own
action rather than a shop slot. `can_buy_tavern_spell` is the mask's question
and `buy_tavern_spell` is the purchase; every refusal in one is a raise in the
other, which is what keeps a legal action from crashing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import src.envs.minibg  # noqa: F401  (breaks a circular import at collection)
from src.bg_catalog.patch_context import PatchContext
from src.bg_lobby.player import PlayerPhase, PlayerState
from src.bg_player_turn.context import PlayerTurnContext
from src.bg_player_turn.engine import PlayerTurnEngine
from src.bg_recruitment import tavern_spells as ts
from src.bg_recruitment.shop_triggers import ShopTriggers
from src.envs.bglike import actions as A

PATCH_DIR = Path("data/bgcore/36_2_0_248348")
HEALTH_SPELL = "BG28_571"  # Hasty Excavation — "costs Health to buy instead of Gold"


@pytest.fixture(scope="module")
def patch() -> PatchContext:
    return PatchContext.load(PATCH_DIR)


def _seat(patch, *, gold=10, tier=4, hero_id=None) -> PlayerState:
    player = PlayerState(
        health=30,
        hero_damage_taken_total=0,
        gold=gold,
        tavern_tier=tier,
        ruleset=patch.meta.ruleset,
        board=[],
        shop=[None] * A.MAX_SHOP_SLOTS,
        hand=[None] * A.HAND_SIZE,
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
    )
    player.round_number = 6
    if hero_id is not None:
        player.hero = patch.heroes[hero_id]
    return player


def _offer(patch, player, card_id=None):
    ts.offer_tavern_spells(
        player,
        rng=np.random.default_rng(0),
        patch=patch,
        card_ids=[card_id] if card_id else None,
    )
    return player.tavern_spell_offers[0]


def _engine(patch):
    rng = np.random.default_rng(0)
    ctx = PlayerTurnContext(
        rng=rng, triggers=ShopTriggers(rng, patch=patch), patch=patch, round_number=6
    )
    return PlayerTurnEngine(A), ctx


# --------------------------------------------------------------------------- #
# The purchase
# --------------------------------------------------------------------------- #


def test_the_spell_moves_from_the_counter_to_the_hand(patch):
    player = _seat(patch)
    spell = _offer(patch, player)
    gold = player.gold
    engine, ctx = _engine(patch)

    assert int(A.Action.BUY_TAVERN_SPELL) in engine.legal_actions(
        player, patch.meta.ruleset
    )
    assert engine.apply(player, int(A.Action.BUY_TAVERN_SPELL), ctx) is True

    assert player.hand[0] is spell
    assert player.tavern_spell_offers == ()
    assert player.gold == gold - spell.cost
    # A purchase is a shop action; the caller spends the budget on a True.
    assert int(A.Action.BUY_TAVERN_SPELL) not in engine.legal_actions(
        player, patch.meta.ruleset
    )


def test_the_mask_and_the_purchase_never_disagree(patch):
    """Every refusal `can_buy_tavern_spell` gives is one `buy_tavern_spell`
    raises — so an action the mask offers cannot crash, and one it withholds
    cannot half-happen."""
    cases = []

    empty = _seat(patch)  # nothing on the counter
    cases.append(empty)

    broke = _seat(patch, gold=0)
    _offer(patch, broke, "BG28_503")
    cases.append(broke)

    full = _seat(patch)
    _offer(patch, full)
    full.hand = [patch.make_minion("BGS_119") for _ in full.hand]
    cases.append(full)

    finished = _seat(patch)
    _offer(patch, finished)
    finished.phase = PlayerPhase.DONE  # buying is a recruit-phase move
    cases.append(finished)

    for player in cases:
        assert not ts.can_buy_tavern_spell(player, 0)
        with pytest.raises(ts.TavernSpellNotAllowed):
            ts.buy_tavern_spell(player, 0, patch=patch)


def test_a_spell_bought_is_a_spell_playable(patch):
    """The whole point of the action: what it puts in hand is a card the seat
    can then play."""
    player = _seat(patch, gold=20)
    _offer(patch, player, "BG28_503")
    engine, ctx = _engine(patch)
    engine.apply(player, int(A.Action.BUY_TAVERN_SPELL), ctx)
    player.board.append(patch.make_minion("BGS_119"))
    player.shop_actions_used = 0
    assert int(A.Action.PLAY_HAND_0) in engine.legal_actions(player, patch.meta.ruleset)


# --------------------------------------------------------------------------- #
# What it costs
# --------------------------------------------------------------------------- #


def test_a_spell_bought_in_health_does_not_ask_for_gold(patch):
    player = _seat(patch, gold=0)
    spell = _offer(patch, player, HEALTH_SPELL)
    assert ts.spell_costs_health(spell)
    assert ts.can_buy_tavern_spell(player, 0)

    player.armor = 1
    health = player.health
    ts.buy_tavern_spell(player, 0, patch=patch)
    # Armor absorbs first, the way it does for any other hero damage.
    assert (player.health, player.armor) == (health - (spell.cost - 1), 0)
    assert player.gold == 0  # buying is not casting; the card pays out when played


def test_the_tavern_does_not_offer_a_purchase_that_would_kill(patch):
    player = _seat(patch, gold=0)
    spell = _offer(patch, player, HEALTH_SPELL)
    player.health, player.armor = spell.cost, 0
    assert not ts.can_buy_tavern_spell(player, 0)
    player.health = spell.cost + 1
    assert ts.can_buy_tavern_spell(player, 0)


def test_taethelans_every_third_spell_is_free(patch):
    """A passive that could never fire before: nothing in either action space
    reached the purchase it counts."""
    hero = next(
        hid for hid, h in patch.heroes.items() if h.name == "Tae'thelan Bloodwatcher"
    )
    player = _seat(patch, gold=99, hero_id=hero)
    paid = []
    for _ in range(6):
        spell = _offer(patch, player)
        before = player.gold
        ts.buy_tavern_spell(player, 0, patch=patch)
        paid.append(before - player.gold)
        player.hand = [None] * len(player.hand)
    assert paid[2] == 0 and paid[5] == 0
    assert all(p > 0 for i, p in enumerate(paid) if i not in (2, 5))


# --------------------------------------------------------------------------- #
# The action space
# --------------------------------------------------------------------------- #


def test_one_offer_one_action(patch):
    """The action names no slot because a tavern shows one spell. A package
    that offered more would leave the extras unbuyable."""
    for package in sorted(Path("data/bgcore").iterdir()):
        if not (package / "meta.json").exists():
            continue
        ctx = PatchContext.load(package)
        assert ctx.meta.ruleset.tavern_spells_per_roll <= 1, package.name


def test_the_structured_path_can_buy_one_too(patch):
    from src.envs.bglike.action_map import struct_action_to_game_action
    from src.envs.minibg.structured_actions import (
        StructAction,
        StructActionType,
        validate_struct_action,
    )

    token = StructAction(StructActionType.BUY_TAVERN_SPELL, ())
    validate_struct_action(token, hand_size=10, board_size=7, max_shop_slots=7)
    assert struct_action_to_game_action(token) == int(A.Action.BUY_TAVERN_SPELL)
    with pytest.raises(ValueError):
        validate_struct_action(
            StructAction(StructActionType.BUY_TAVERN_SPELL, (0,)),
            hand_size=10,
            board_size=7,
            max_shop_slots=7,
        )


def test_the_classic_packages_have_no_spell_to_buy(patch):
    """They carry no Tavern spells at all, so the action is never legal there
    and a 2021 policy sees exactly the mask it was trained on."""
    engine, _ = _engine(patch)
    for package in ("data/bgcore/19_6_0_74257", "data/bgcore/15_6_2_36393"):
        ctx = PatchContext.load(Path(package))
        assert ts.tavern_spell_pool(6, patch=ctx) == [], package
        player = _seat(ctx)
        assert not ts.can_buy_tavern_spell(player, 0)
        assert int(A.Action.BUY_TAVERN_SPELL) not in engine.legal_actions(
            player, ctx.meta.ruleset
        )
