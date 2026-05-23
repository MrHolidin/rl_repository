"""League sampler and scripted slot tests."""

from __future__ import annotations

import random

from src.training.selfplay.game_record import (
    SLOT_CURRENT,
    build_scripted_slot_map,
    invert_scripted_slot_map,
)
from src.training.selfplay.league_config import parse_league_settings
from src.training.selfplay.league_config import LeagueSamplerConfig
from src.training.selfplay.league_sampler import LeagueSyncState, sample_league_opponent


def test_build_scripted_slot_map_unique_negative_ids():
    mapping = build_scripted_slot_map(["structured", "t1_random", "t_up_random"])
    assert mapping == {
        "structured": -2,
        "t1_random": -3,
        "t_up_random": -4,
    }
    assert invert_scripted_slot_map(mapping) == {
        -2: "structured",
        -3: "t1_random",
        -4: "t_up_random",
    }


def test_sample_league_opponent_uses_per_key_scripted_slot():
    scripted = {"t1_random": 0.5, "structured": 0.5}
    slot_ids = build_scripted_slot_map(scripted.keys())
    sync = LeagueSyncState(frozen_pool={}, win_rates={sid: 0.5 for sid in slot_ids.values()})
    sampler = LeagueSamplerConfig(kind="fractional", current_self_fraction=0.0, past_self_fraction=0.0)
    rng = random.Random(0)
    seen = set()
    for _ in range(40):
        sample = sample_league_opponent(
            game_rng=rng,
            use_self_play=True,
            sync=sync,
            sampler=sampler,
            scripted_distribution=scripted,
            scripted_slot_ids=slot_ids,
            slot_id_to_scripted_key=invert_scripted_slot_map(slot_ids),
            frozen_pool={},
        )
        seen.add(sample.slot_id)
    assert seen <= {-2, -3}


def test_pfsp_unified_draws_from_scripted_and_frozen():
    scripted = {"t1_random": 1.0}
    slot_ids = build_scripted_slot_map(scripted.keys())
    frozen = {0: b"x", 1: b"y"}
    sync = LeagueSyncState(
        frozen_pool=frozen,
        win_rates={-2: 0.2, 0: 0.5, 1: 0.8, SLOT_CURRENT: 0.5},
    )
    sampler = LeagueSamplerConfig(kind="pfsp_unified", current_self_fraction=0.0)
    rng = random.Random(1)
    seen = set()
    for _ in range(50):
        sample = sample_league_opponent(
            game_rng=rng,
            use_self_play=True,
            sync=sync,
            sampler=sampler,
            scripted_distribution=scripted,
            scripted_slot_ids=slot_ids,
            slot_id_to_scripted_key=invert_scripted_slot_map(slot_ids),
            frozen_pool=frozen,
        )
        seen.add(sample.slot_id)
    assert 0 in seen or 1 in seen
    assert -2 in seen


def test_parse_league_settings_nested_yaml():
    settings = parse_league_settings(
        {
            "league": {
                "rating": {"kind": "trueskill", "trueskill": {"draw_probability": 0.0}},
                "sampler": {"kind": "pfsp_unified", "current_self_fraction": 0.1},
            },
            "self_play": {"max_frozen_agents": 10},
        }
    )
    assert settings.rating.kind == "trueskill"
    assert settings.sampler.kind == "pfsp_unified"
    assert settings.sampler.current_self_fraction == 0.1


def test_trueskill_quality_sampler_uses_match_quality(monkeypatch):
    from src.training.selfplay import league_sampler as ls

    calls: list[str] = []

    def _fake_quality(pool_ids, *, sync, game_rng):
        calls.append("quality")
        return pool_ids[0]

    monkeypatch.setattr(ls, "_pick_by_match_quality", _fake_quality)

    scripted = {"t1_random": 1.0}
    slot_ids = build_scripted_slot_map(scripted.keys())
    sync = LeagueSyncState(
        frozen_pool={0: b"x"},
        win_rates={-2: 0.5, 0: 0.5, SLOT_CURRENT: 0.5},
        rating_kind="trueskill",
        trueskill={},
    )
    sampler = LeagueSamplerConfig(kind="trueskill_quality", current_self_fraction=0.0)
    sample = sample_league_opponent(
        game_rng=random.Random(0),
        use_self_play=True,
        sync=sync,
        sampler=sampler,
        scripted_distribution=scripted,
        scripted_slot_ids=slot_ids,
        slot_id_to_scripted_key=invert_scripted_slot_map(slot_ids),
        frozen_pool={0: b"x"},
    )
    assert calls == ["quality"]
    assert sample.slot_id in {-2, 0}


def test_trueskill_quality_requires_trueskill_rating():
    import pytest

    with pytest.raises(ValueError, match="trueskill_quality requires"):
        parse_league_settings(
            {
                "league": {
                    "rating": {"kind": "ema"},
                    "sampler": {"kind": "trueskill_quality"},
                }
            }
        )
