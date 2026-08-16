"""Characterisation snapshot of the observation / action **layout**.

The layout (slot field offsets, race and tier vocabularies, action indices,
observation widths) is what a trained checkpoint is wired to, and today it lives
in module-level constants shared by every patch package. Making it per-patch
data — so a new client build can be added without touching code — is only safe
if the numbers the *existing* packages produce stay exactly where they are.

This test pins those numbers. It is deliberately dumb: it recomputes the layout,
serialises it, and compares against ``tests/fixtures/patch_layout_snapshot.json``.
Any reordering, widening or renaming shows up as a diff, per package.

Regenerating (only when the change is intended)::

    UPDATE_LAYOUT_SNAPSHOT=1 pytest tests/test_patch_layout_snapshot.py -k update

then commit the fixture alongside the change that caused it.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from src.bg_catalog.patch_context import PatchContext, load_patch_context
from src.bg_catalog.ruleset import Ruleset
from src.envs.bglike import action_map as bglike_action_map
from src.envs.bglike import actions as bglike_actions
from src.envs.bglike import obs as bglike_obs
from src.envs.bglike import obs_v5, obs_v5_heroes, obs_v6_heroes, obs_v7_pref
from src.envs.bglike.game import BGLikeGame
from src.envs.bglike.tribe_pref import NUM_TRIBES, TRIBES
from src.envs.minibg import obs as minibg_obs

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "patch_layout_snapshot.json"

# Every package under data/bgcore is snapshotted; a new one has to be added to
# the fixture consciously (the test fails on an unknown package until then).
PACKAGES_DIR = REPO_ROOT / "data" / "bgcore"

# The five bglike observation layouts a config can select via ``obs_kind``.
OBS_BUILDERS = {
    "bglike": (bglike_obs.OBS_DIM, bglike_obs.build_observation),
    "bglike_v5": (obs_v5.OBS_DIM_V5, obs_v5.build_observation_v5),
    "bglike_v5_heroes": (obs_v5_heroes.OBS_DIM_V5_HEROES, obs_v5_heroes.build_observation_v5_heroes),
    "bglike_v6_heroes": (obs_v6_heroes.OBS_DIM_V6_HEROES, obs_v6_heroes.build_observation_v6_heroes),
    "bglike_v7_pref": (obs_v7_pref.OBS_DIM_V7_PREF, obs_v7_pref.build_observation_v7_pref),
}

# Fixed inputs for the observation fingerprints below.
PROBE_SEED = 0
PROBE_SEAT = 0


def _packages() -> list[Path]:
    return sorted(p for p in PACKAGES_DIR.iterdir() if (p / "meta.json").is_file())


def _slot_layout() -> Dict[str, Any]:
    """Field offsets inside one minion slot — shared by minibg and bglike."""
    return {
        "SLOT_DIM": int(minibg_obs.SLOT_DIM),
        "PRESENCE_OFFSET": int(minibg_obs.PRESENCE_OFFSET),
        "CARD_IDX_OFFSET": int(minibg_obs.CARD_IDX_OFFSET),
        "TIER_OFFSET": int(minibg_obs.TIER_OFFSET),
        "STATS_OFFSET": int(minibg_obs.STATS_OFFSET),
        "RACE_OFFSET": int(minibg_obs.RACE_OFFSET),
        "KEYWORD_OFFSET": int(minibg_obs.KEYWORD_OFFSET),
        "SHIELD_OFFSET": int(minibg_obs.SHIELD_OFFSET),
        "GOLDEN_OFFSET": int(minibg_obs.GOLDEN_OFFSET),
        "NUM_KEYWORD_CHANNELS": int(minibg_obs.NUM_KEYWORD_CHANNELS),
        "NUM_TRIGGER_CHANNELS": int(minibg_obs.NUM_TRIGGER_CHANNELS),
        "NUM_EFFECT_CHANNELS": int(minibg_obs.NUM_EFFECT_CHANNELS),
    }


def _ability_token_vocabularies() -> Dict[str, Any]:
    """Id-space sizes of the v5 ability tokens.

    These size embedding tables inside the networks rather than the observation
    vector, so they do not show up in any width — and an engine change that grew
    one (``len(Keyword) + 1`` did exactly that) would invalidate every v5-family
    checkpoint without moving a single number the rest of this snapshot pins.
    """
    return {
        "NUM_EFFECT_IDS": int(obs_v5.NUM_EFFECT_IDS),
        "NUM_TRIGGER_IDS": int(obs_v5.NUM_TRIGGER_IDS),
        "NUM_CONDITION_KIND_IDS": int(obs_v5.NUM_CONDITION_KIND_IDS),
        "NUM_RACE_IDS": int(obs_v5.NUM_RACE_IDS),
        "NUM_KEYWORD_IDS": int(obs_v5.NUM_KEYWORD_IDS),
        "unencoded_effects": sorted(minibg_obs.UNENCODED_EFFECTS),
    }


def _vocabularies() -> Dict[str, Any]:
    """Race / tier vocabularies — the first things a modern package widens."""
    return {
        "race_order": [r.name if r is not None else "NONE" for r in minibg_obs._RACE_ORDER],
        "RACE_ONEHOT_DIM": int(minibg_obs.RACE_ONEHOT_DIM),
        "NUM_TIER_ONEHOT": int(minibg_obs.NUM_TIER_ONEHOT),
        "tribe_pref_tribes": [r.name for r in TRIBES],
        "tribe_pref_num_tribes": int(NUM_TRIBES),
    }


def _action_layout() -> Dict[str, Any]:
    """Action indices, by name, in enum order — catches any insertion."""
    names_by_index = [a.name for a in sorted(bglike_actions.Action, key=lambda a: int(a))]
    return {
        "BOARD_SIZE": int(bglike_actions.BOARD_SIZE),
        "HAND_SIZE": int(bglike_actions.HAND_SIZE),
        "MAX_SHOP_SLOTS": int(bglike_actions.MAX_SHOP_SLOTS),
        "MAX_TIER": int(bglike_actions.MAX_TIER),
        "SHOP_OFFERS_BY_TIER": {str(k): int(v) for k, v in sorted(bglike_actions.SHOP_OFFERS_BY_TIER.items())},
        "NUM_ACTIONS": int(bglike_actions.NUM_ACTIONS),
        "NUM_ENV_ACTIONS": int(bglike_action_map.NUM_ENV_ACTIONS),
        "A_SWAP_BOARD_0": int(bglike_action_map.A_SWAP_BOARD_0),
        "A_APPLY_EFFECT_SKIP": int(bglike_action_map.A_APPLY_EFFECT_SKIP),
        "game_actions_by_index": names_by_index,
    }


def _obs_dims() -> Dict[str, int]:
    dims = {"minibg": int(minibg_obs.OBS_DIM)}
    for kind, (dim, _builder) in OBS_BUILDERS.items():
        dims[kind] = int(dim)
    return dims


def _ruleset_dict(rs: Ruleset) -> Dict[str, Any]:
    return {
        "max_tier": int(rs.max_tier),
        "level_up_costs": {str(k): int(v) for k, v in sorted(rs.level_up_costs.items())},
        "level_up_discount_per_round": int(rs.level_up_discount_per_round),
        "gold_per_round": {str(k): int(v) for k, v in sorted(rs.gold_per_round.items())},
        "gold_cap": int(rs.gold_cap),
        "buy_cost": int(rs.buy_cost),
        "sell_reward": int(rs.sell_reward),
        "roll_cost": int(rs.roll_cost),
        "starting_health": int(rs.starting_health),
        "damage_cap_schedule": [
            [None if step.max_round is None else int(step.max_round), int(step.cap)]
            for step in rs.damage_cap_schedule
        ],
        "damage_cap_lifted_at_alive": int(rs.damage_cap_lifted_at_alive),
        "max_rounds": int(rs.max_rounds),
    }


def _layout_dict(layout) -> Dict[str, Any]:
    return {
        "races": [r.name for r in layout.races],
        "race_onehot_dim": int(layout.race_onehot_dim),
        "num_tiers": int(layout.num_tiers),
        "max_shop_slots": int(layout.max_shop_slots),
        "shop_offers_by_tier": {str(k): int(v) for k, v in sorted(layout.shop_offers_by_tier.items())},
        "discover_picks": int(layout.discover_picks),
    }


def _obs_fingerprints(patch_dir: Path) -> Dict[str, Any]:
    """Hash of each layout's observation vector on a fixed seed and seat.

    Widths alone would not catch a field moving *within* the vector; the hash
    does. It is content-sensitive by design, so a deliberate change to what the
    observation encodes (or to the opening shop roll) requires regenerating the
    fixture — which is exactly the moment to check no layout drifted with it.
    """
    game = BGLikeGame(seed=PROBE_SEED, patch_dir=str(patch_dir))
    state = game.initial_state()
    out: Dict[str, Any] = {}
    for kind, (dim, builder) in OBS_BUILDERS.items():
        vec = builder(
            state,
            PROBE_SEAT,
            0.0,
            is_my_turn=True,
            patch=game._patch,
        )
        arr = np.ascontiguousarray(vec, dtype=np.float32)
        assert arr.shape == (dim,), f"{kind}: obs width {arr.shape[0]} != declared {dim}"
        out[kind] = {
            "shape": int(arr.shape[0]),
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        }
    return out


def _package_snapshot(patch_dir: Path) -> Dict[str, Any]:
    ctx: PatchContext = load_patch_context(str(patch_dir))
    return {
        "build": int(ctx.build),
        "patch": str(ctx.patch),
        "num_pool_indices": int(ctx.num_pool_indices),
        "num_templates": len(ctx.templates),
        "num_pool_ids": len(ctx.pool_ids),
        "rotation_tribes": [r.name for r in ctx.meta.rotation_tribes],
        "rotation_excluded_count": int(ctx.meta.rotation_excluded_count),
        "pool_copies_by_tier": {str(k): int(v) for k, v in sorted(ctx.meta.pool_copies_by_tier.items())},
        "ruleset": _ruleset_dict(ctx.meta.ruleset),
        "layout": _layout_dict(ctx.meta.layout),
        "obs_fingerprints": _obs_fingerprints(patch_dir),
    }


def build_snapshot() -> Dict[str, Any]:
    return {
        "slot_layout": _slot_layout(),
        "vocabularies": _vocabularies(),
        "ability_token_vocabularies": _ability_token_vocabularies(),
        "action_layout": _action_layout(),
        "obs_dims": _obs_dims(),
        "packages": {p.name: _package_snapshot(p) for p in _packages()},
    }


def _load_fixture() -> Dict[str, Any]:
    with FIXTURE.open(encoding="utf-8") as f:
        return json.load(f)


def _flatten(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """Leaf paths of a nested dict, so a mismatch names the exact field."""
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    return {prefix: obj}


def test_layout_snapshot_matches_fixture():
    actual = _flatten(build_snapshot())
    expected = _flatten(_load_fixture())

    missing = sorted(set(expected) - set(actual))
    added = sorted(set(actual) - set(expected))
    changed = {
        key: (expected[key], actual[key])
        for key in sorted(set(actual) & set(expected))
        if actual[key] != expected[key]
    }

    assert not missing, f"layout fields disappeared: {missing}"
    assert not added, f"unpinned layout fields appeared: {added}"
    assert not changed, "layout drifted (expected → actual): " + json.dumps(
        {k: {"expected": e, "actual": a} for k, (e, a) in changed.items()}, indent=2
    )


def test_declared_obs_dims_match_built_vectors():
    """The width a config sizes the network from is the width the env emits."""
    dims = _obs_dims()
    for patch_dir in _packages():
        for kind, entry in _obs_fingerprints(patch_dir).items():
            assert entry["shape"] == dims[kind], (
                f"{patch_dir.name}/{kind}: built {entry['shape']} != declared {dims[kind]}"
            )


def test_obs_fingerprints_are_deterministic():
    """A fingerprint that varies run to run would pin nothing."""
    patch_dir = _packages()[0]
    assert _obs_fingerprints(patch_dir) == _obs_fingerprints(patch_dir)


@pytest.mark.skipif(
    os.environ.get("UPDATE_LAYOUT_SNAPSHOT") != "1",
    reason="set UPDATE_LAYOUT_SNAPSHOT=1 to regenerate the fixture",
)
def test_update_layout_snapshot():
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE.open("w", encoding="utf-8") as f:
        json.dump(build_snapshot(), f, indent=2, sort_keys=True)
        f.write("\n")
