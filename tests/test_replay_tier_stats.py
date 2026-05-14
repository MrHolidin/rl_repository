"""MiniBG replay tier aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.envs.minibg.replay_tier_stats import analyze_replay_file, aggregate_paths
from src.training.trainer import StartPolicy
from src.utils.match import resolve_opening_agent_token


def test_resolve_opening_agent_token_matches_random_coin() -> None:
    seed = 70406
    rng = __import__("random").Random(seed)
    expected = 1 if rng.random() < 0.5 else -1
    assert resolve_opening_agent_token(StartPolicy.RANDOM, seed=seed) == expected


def test_analyze_replay_requires_learned_player(tmp_path: Path) -> None:
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps({"type": "header", "format": 2, "game": "minibg"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="learned_player_index"):
        analyze_replay_file(p)


def test_analyze_minimal_tier_milestone(tmp_path: Path) -> None:
    p = tmp_path / "m.jsonl"
    hdr = {
        "type": "header",
        "format": 2,
        "game": "minibg",
        "learned_player_index": 0,
        "scripted_player_index": 1,
        "learned_agent_kind": "test_learned",
        "scripted_opponent": "test_bot",
    }

    def frame(round_: int, t0: int, t1: int) -> dict:
        return {
            "type": "frame",
            "ep": 1,
            "i": 1,
            "p": 0,
            "a": 0,
            "illegal": False,
            "state": {
                "round": round_,
                "cur": 0,
                "init": 0,
                "done": False,
                "winner": None,
                "shop_excluded_race": None,
                "p0": {"tier": t0},
                "p1": {"tier": t1},
            },
            "info": {},
        }

    lines = [
        json.dumps(hdr),
        json.dumps(frame(1, 1, 1)),
        json.dumps(frame(2, 2, 2)),
        json.dumps(frame(3, 2, 3)),
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _, lo, so = analyze_replay_file(p)
    assert lo.first_round_at_tier[2] == 2
    assert lo.first_round_at_tier[3] is None
    assert so.first_round_at_tier[2] == 2
    assert so.first_round_at_tier[3] == 3
    assert lo.final_tier == 2
    assert so.final_tier == 3

    agg = aggregate_paths([p])
    assert agg["learned:test_learned"].games == 1
    assert agg["learned:test_learned"].first_round_at_tier[2] == [2]
