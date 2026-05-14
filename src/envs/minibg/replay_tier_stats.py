"""Tier milestones from MiniBG JSONL replays (needs ``learned_player_index`` in header)."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple

from src.envs.minibg.actions import MAX_TIER


def load_header(path: Path) -> Dict[str, Any]:
    line = path.read_text(encoding="utf-8").splitlines()[0]
    rec = json.loads(line)
    if rec.get("type") != "header":
        raise ValueError(f"{path}: first record is not header")
    return rec


def iter_frames(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") == "frame":
                yield rec


@dataclass
class RoleTierOutcome:
    role_label: str
    first_round_at_tier: Dict[int, Optional[int]]
    final_tier: int


@dataclass
class TierStatsRecord:
    """Per-role aggregates built from many replay files."""

    games: int = 0
    first_round_at_tier: DefaultDict[int, List[int]] = field(
        default_factory=lambda: defaultdict(list)
    )
    final_tier: List[int] = field(default_factory=list)


def _role_labels(header: Dict[str, Any]) -> Tuple[str, str]:
    lk = str(header.get("learned_agent_kind") or "learned_unknown")
    sk = str(header.get("scripted_opponent") or header.get("opponent") or "scripted_unknown")
    return f"learned:{lk}", f"scripted:{sk}"


def analyze_replay_file(path: Path) -> Tuple[Dict[str, Any], RoleTierOutcome, RoleTierOutcome]:
    header = load_header(path)
    learned_pi = header.get("learned_player_index")
    if learned_pi is None:
        raise ValueError(
            f"{path}: header missing learned_player_index — re-record replay with current eval."
        )
    learned_pi = int(learned_pi)
    scripted_pi = int(header.get("scripted_player_index", 1 - learned_pi))

    prev_tier = {0: 1, 1: 1}
    first_round: Dict[Tuple[int, int], int] = {}

    for row in iter_frames(path):
        st = row["state"]
        rnd = int(st["round"])
        for p in (0, 1):
            pl = st.get(f"p{p}") or {}
            tier = int(pl.get("tier", 1))
            pt = prev_tier[p]
            if tier > pt:
                for tnew in range(pt + 1, tier + 1):
                    key = (p, tnew)
                    if key not in first_round:
                        first_round[key] = rnd
                prev_tier[p] = tier

    rl, rs = _role_labels(header)
    fd_learned = prev_tier[learned_pi]
    fd_scripted = prev_tier[scripted_pi]

    learned_out = RoleTierOutcome(
        rl,
        {t: first_round.get((learned_pi, t)) for t in range(2, MAX_TIER + 1)},
        fd_learned,
    )
    scripted_out = RoleTierOutcome(
        rs,
        {t: first_round.get((scripted_pi, t)) for t in range(2, MAX_TIER + 1)},
        fd_scripted,
    )
    return header, learned_out, scripted_out


def merge_outcome(agg: DefaultDict[str, TierStatsRecord], out: RoleTierOutcome) -> None:
    rec = agg[out.role_label]
    rec.games += 1
    rec.final_tier.append(out.final_tier)
    for tier, rnd in out.first_round_at_tier.items():
        if rnd is not None:
            rec.first_round_at_tier[tier].append(rnd)


def aggregate_paths(paths: List[Path]) -> Dict[str, TierStatsRecord]:
    agg: DefaultDict[str, TierStatsRecord] = defaultdict(TierStatsRecord)
    for p in paths:
        _, lo, so = analyze_replay_file(p)
        merge_outcome(agg, lo)
        merge_outcome(agg, so)
    return dict(agg)


def format_report(agg: Dict[str, TierStatsRecord]) -> str:
    lines: List[str] = []
    for role in sorted(agg.keys()):
        rec = agg[role]
        lines.append(f"=== {role} ({rec.games} games) ===")
        if rec.final_tier:
            m = sum(rec.final_tier) / len(rec.final_tier)
            lines.append(f"  final tavern tier (mean): {m:.2f}")
        for tier in range(2, MAX_TIER + 1):
            xs = rec.first_round_at_tier.get(tier, [])
            reached = len(xs)
            freq = reached / rec.games if rec.games else 0.0
            if xs:
                avg_r = sum(xs) / len(xs)
                lines.append(
                    f"  tier {tier}: reached in {100.0 * freq:.0f}% games "
                    f"(mean first round {avg_r:.2f}; n={reached})"
                )
            else:
                lines.append(f"  tier {tier}: never reached ({freq:.0%})")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "RoleTierOutcome",
    "TierStatsRecord",
    "aggregate_paths",
    "analyze_replay_file",
    "format_report",
    "iter_frames",
    "load_header",
]
