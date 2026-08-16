"""The layout seam: ``PatchLayout`` must describe what the encoders actually do.

``PatchLayout`` is the data-side declaration of the vocabularies the observation
encoders and the action space are built from. Until every call site reads it,
the constants remain the truth and the dataclass merely mirrors them — so the
mirror is what these tests check. Once a call site is converted, its constant
becomes a derived value and the corresponding assertion here becomes trivially
true; that is the intended direction of travel, not a reason to drop the test.

The failure mode this rules out: a modern package widens the layout, the
encoder keeps using its own constant, and every observation is quietly encoded
against the wrong vocabulary.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.bg_catalog.layout import (
    DEFAULT_LAYOUT,
    LayoutValidationError,
    PatchLayout,
    layout_from_meta,
    validate_layout,
)
from src.bg_catalog.patch_context import load_patch_context
from src.bg_core.minion import Race
from src.envs.bglike import actions as bglike_actions
from src.envs.minibg import obs as minibg_obs

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGES_DIR = _REPO_ROOT / "data" / "bgcore"


def _packages():
    return sorted(p for p in _PACKAGES_DIR.iterdir() if (p / "meta.json").is_file())


# --------------------------------------------------------------------------- #
# The default layout mirrors today's constants exactly.
# --------------------------------------------------------------------------- #


def test_default_race_order_matches_encoder():
    assert DEFAULT_LAYOUT.race_order == minibg_obs._RACE_ORDER
    assert DEFAULT_LAYOUT.race_onehot_dim == minibg_obs.RACE_ONEHOT_DIM


def test_default_race_index_matches_encoder_positions():
    for race in DEFAULT_LAYOUT.race_order:
        assert DEFAULT_LAYOUT.race_index(race) == minibg_obs._RACE_ORDER.index(race)


def test_default_tier_width_matches_encoder():
    assert DEFAULT_LAYOUT.num_tiers == minibg_obs.NUM_TIER_ONEHOT


def test_default_shop_layout_matches_action_space():
    assert DEFAULT_LAYOUT.max_shop_slots == bglike_actions.MAX_SHOP_SLOTS
    assert dict(DEFAULT_LAYOUT.shop_offers_by_tier) == dict(bglike_actions.SHOP_OFFERS_BY_TIER)
    for tier in range(1, bglike_actions.MAX_TIER + 1):
        assert DEFAULT_LAYOUT.shop_offers(tier) == bglike_actions.shop_offers_count(tier)


def test_default_discover_picks_matches_action_space():
    picks = sum(
        1 for a in bglike_actions.Action if a.name.startswith("DISCOVER_PICK_")
    )
    assert DEFAULT_LAYOUT.discover_picks == picks


# --------------------------------------------------------------------------- #
# Shipped packages get the default layout, and their cards fit it.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("patch_dir", _packages(), ids=lambda p: p.name)
def test_shipped_packages_use_the_default_layout(patch_dir):
    """No package declares a layout yet, so all of them must land on the default."""
    assert "layout" not in json.loads((patch_dir / "meta.json").read_text())
    assert load_patch_context(str(patch_dir)).meta.layout == DEFAULT_LAYOUT


@pytest.mark.parametrize("patch_dir", _packages(), ids=lambda p: p.name)
def test_shipped_package_cards_fit_their_layout(patch_dir):
    """Loading already validates; this states it as its own expectation."""
    ctx = load_patch_context(str(patch_dir))
    validate_layout(
        ctx.meta.layout,
        max_tier=ctx.meta.ruleset.max_tier,
        rotation_tribes=ctx.meta.rotation_tribes,
        card_races=[m.race for m in ctx.templates.values()],
        card_tiers=[m.tier for m in ctx.templates.values()],
        package=patch_dir.name,
    )


# --------------------------------------------------------------------------- #
# Engine additions must not resize what a checkpoint is wired to.
# --------------------------------------------------------------------------- #


def test_a_new_keyword_does_not_resize_the_ability_vocabulary():
    from src.bg_core.effects import Keyword
    from src.envs.bglike.obs_v5 import NUM_KEYWORD_IDS, _keyword_id

    assert len(Keyword) + 1 > NUM_KEYWORD_IDS, (
        "the engine should know keywords this frozen vocabulary does not encode; "
        "if it does not, this test has stopped testing anything"
    )
    assert _keyword_id(Keyword.VENOMOUS) == 0, "outside the vocabulary → unknown id"
    assert _keyword_id(Keyword.REBORN) == int(Keyword.REBORN.value)


def test_a_new_effect_does_not_resize_the_ability_vocabulary():
    from src.bg_core.effects import AvengeEffect, BuffSelf
    from src.envs.bglike.obs_v5 import _effect_id

    assert _effect_id(AvengeEffect(count=2, effect=BuffSelf(attack=1, health=1))) == 0
    assert _effect_id(BuffSelf(attack=1, health=1)) > 0


def test_an_unregistered_effect_is_still_loud():
    """The blind spot is opt-in: forgetting to register one must still fail."""
    from dataclasses import dataclass

    from src.envs.bglike.obs_v5 import _effect_id

    @dataclass(frozen=True)
    class NotRegisteredEffect:
        amount: int = 1

    with pytest.raises(KeyError, match="NotRegisteredEffect"):
        _effect_id(NotRegisteredEffect())


# --------------------------------------------------------------------------- #
# Parsing.
# --------------------------------------------------------------------------- #


def test_missing_block_is_the_default():
    assert layout_from_meta(None) is DEFAULT_LAYOUT
    assert layout_from_meta({}) is DEFAULT_LAYOUT


def test_partial_block_keeps_default_fields():
    layout = layout_from_meta({"num_tiers": 7})
    assert layout.num_tiers == 7
    assert layout.races == DEFAULT_LAYOUT.races
    assert layout.max_shop_slots == DEFAULT_LAYOUT.max_shop_slots


def test_races_parse_in_declared_order():
    layout = layout_from_meta({"races": ["BEAST", "MURLOC", "ALL"]})
    assert layout.races == (Race.BEAST, Race.MURLOC, Race.ALL)
    assert layout.race_order == (None, Race.BEAST, Race.MURLOC, Race.ALL)
    assert layout.race_onehot_dim == 4


def test_unknown_race_name_is_rejected():
    with pytest.raises(LayoutValidationError, match="NOT_A_TRIBE"):
        layout_from_meta({"races": ["BEAST", "NOT_A_TRIBE"]})


def test_post_19_6_tribes_parse():
    """Quilboar/Naga/Undead exist in the engine but in no shipped package."""
    layout = layout_from_meta({"races": ["QUILBOAR", "NAGA", "UNDEAD"]})
    assert layout.races == (Race.QUILBOAR, Race.NAGA, Race.UNDEAD)


def test_new_tribes_are_absent_from_the_default_layout():
    """Adding a tribe to Race must not widen what old packages encode."""
    for race in (Race.QUILBOAR, Race.NAGA, Race.UNDEAD):
        assert race not in DEFAULT_LAYOUT.races
    assert DEFAULT_LAYOUT.race_onehot_dim == 9


def test_default_layout_rejects_a_card_from_a_new_tribe():
    """The guard that would catch a modern catalog loaded on a classic layout."""
    with pytest.raises(LayoutValidationError, match="QUILBOAR"):
        _validate(DEFAULT_LAYOUT, card_races=[Race.BEAST, Race.QUILBOAR])


# --------------------------------------------------------------------------- #
# Validation — the checks that make a data-driven layout safe.
# --------------------------------------------------------------------------- #


def _validate(layout: PatchLayout, **overrides):
    kwargs = {
        "max_tier": 6,
        "rotation_tribes": (Race.BEAST,),
        "card_races": [Race.BEAST, None],
        "card_tiers": [1, 6],
    }
    kwargs.update(overrides)
    validate_layout(layout, **kwargs)


def test_card_race_outside_layout_is_rejected():
    layout = PatchLayout(races=(Race.BEAST,))
    with pytest.raises(LayoutValidationError, match="MURLOC"):
        _validate(layout, card_races=[Race.BEAST, Race.MURLOC])


def test_rotation_tribe_outside_layout_is_rejected():
    layout = PatchLayout(races=(Race.BEAST,))
    with pytest.raises(LayoutValidationError, match="DRAGON"):
        _validate(layout, rotation_tribes=(Race.BEAST, Race.DRAGON))


def test_card_tier_past_the_onehot_is_rejected():
    with pytest.raises(LayoutValidationError, match=r"\[7\]"):
        _validate(DEFAULT_LAYOUT, card_tiers=[1, 7])


def test_tier_zero_is_accepted():
    """The triple-reward spell has no tavern tier; an all-zero one-hot is right."""
    _validate(DEFAULT_LAYOUT, card_tiers=[0, 1, 6])


def test_tier_onehot_narrower_than_max_tier_is_rejected():
    with pytest.raises(LayoutValidationError, match="num_tiers"):
        _validate(PatchLayout(num_tiers=6), max_tier=7)


def test_shop_offers_over_slot_count_is_rejected():
    layout = PatchLayout(shop_offers_by_tier={**dict(DEFAULT_LAYOUT.shop_offers_by_tier), 6: 7})
    with pytest.raises(LayoutValidationError, match="exceeds max_shop_slots"):
        _validate(layout)


def test_tier_without_shop_offers_entry_is_rejected():
    layout = PatchLayout(num_tiers=7, shop_offers_by_tier=dict(DEFAULT_LAYOUT.shop_offers_by_tier))
    with pytest.raises(LayoutValidationError, match=r"no entry for tiers \[7\]"):
        _validate(layout, max_tier=7)


def test_duplicate_race_is_rejected():
    layout = PatchLayout(races=(Race.BEAST, Race.BEAST))
    with pytest.raises(LayoutValidationError, match="repeats"):
        _validate(layout)


# --------------------------------------------------------------------------- #
# A modern-shaped layout parses and validates end to end.
# --------------------------------------------------------------------------- #


def test_a_widened_layout_round_trips():
    """What a modern package will declare, minus the races the engine lacks."""
    raw = {
        "races": ["BEAST", "DEMON", "MECHANICAL", "MURLOC", "DRAGON", "PIRATE", "ELEMENTAL", "ALL"],
        "num_tiers": 7,
        "max_shop_slots": 7,
        "shop_offers_by_tier": {"1": 3, "2": 4, "3": 4, "4": 5, "5": 5, "6": 6, "7": 7},
        "discover_picks": 4,
    }
    layout = layout_from_meta(raw)
    assert layout.num_tiers == 7
    assert layout.shop_offers(7) == 7
    assert layout.discover_picks == 4
    _validate(layout, max_tier=7, card_tiers=[1, 7])
