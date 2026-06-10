"""Tests for bglike forced_tribes / excluded_tribes config resolution."""

from __future__ import annotations

import pytest

from src.bg_catalog.patch_context import load_patch_context
from src.bg_core.minion import Race
from src.envs.bglike.game import BGLikeGame
from src.envs.bglike.lobby_env import make_bglike_training_env
from src.envs.bglike.tribe_config import (
    apply_tribe_params_to_lobby_kwargs,
    resolve_shop_excluded_races,
)

_PATCH_74257 = "data/bgcore/19_6_0_74257"
_PATCH_36393 = "data/bgcore/15_6_2_36393"


def test_excluded_tribes_maps_to_shop_excluded_race():
    patch = load_patch_context(_PATCH_74257)
    excluded = resolve_shop_excluded_races(
        patch=patch,
        excluded_tribes=["DRAGON", "PIRATE", "ELEMENTAL"],
    )
    assert excluded == (Race.DRAGON, Race.PIRATE, Race.ELEMENTAL)


def test_forced_tribes_computes_complement():
    patch = load_patch_context(_PATCH_36393)
    excluded = resolve_shop_excluded_races(
        patch=patch,
        forced_tribes=["MURLOC", "BEAST", "DEMON"],
    )
    assert excluded == (Race.MECHANICAL,)


def test_forced_and_excluded_both_set_raises():
    patch = load_patch_context(_PATCH_36393)
    with pytest.raises(ValueError, match="not both"):
        resolve_shop_excluded_races(
            patch=patch,
            forced_tribes=["MURLOC"],
            excluded_tribes=["BEAST"],
        )


def test_apply_tribe_params_to_lobby_kwargs():
    out = apply_tribe_params_to_lobby_kwargs(
        {
            "patch_dir": _PATCH_74257,
            "excluded_tribes": ["DRAGON", "PIRATE", "ELEMENTAL"],
            "seed": 1,
        }
    )
    assert "excluded_tribes" not in out
    assert "forced_tribes" not in out
    assert out["shop_excluded_race"] == (
        Race.DRAGON,
        Race.PIRATE,
        Race.ELEMENTAL,
    )


def test_training_env_factory_stores_excluded_tribes():
    env = make_bglike_training_env(
        (0,),
        patch_dir=_PATCH_74257,
        excluded_tribes=["DRAGON", "PIRATE", "ELEMENTAL"],
        seed=7,
    )
    assert env._shop_excluded_race == (
        Race.DRAGON,
        Race.PIRATE,
        Race.ELEMENTAL,
    )


def test_game_direct_constructor_still_accepts_shop_excluded_race():
    game = BGLikeGame(
        seed=0,
        patch_dir=_PATCH_36393,
        shop_excluded_race=Race.MURLOC,
    )
    state = game.initial_state()
    assert state.shop_excluded_race == (Race.MURLOC,)
