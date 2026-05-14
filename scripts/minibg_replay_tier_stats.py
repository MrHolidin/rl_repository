#!/usr/bin/env python3
"""Aggregate MiniBG JSONL tier milestones (requires ``learned_player_index`` in replay header).

Example::

    python3 scripts/minibg_replay_tier_stats.py runs/minibg/my_run/replays_vs_*/*.jsonl
"""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.minibg.replay_tier_stats import aggregate_paths, format_report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "jsonl",
        nargs="+",
        type=pathlib.Path,
        help="Replay file paths (*.jsonl)",
    )
    args = ap.parse_args()
    paths = [p.resolve() for p in args.jsonl if p.suffix.lower() == ".jsonl"]
    if not paths:
        print("No .jsonl paths given.", file=sys.stderr)
        sys.exit(1)
    missing = [p for p in paths if not p.is_file()]
    if missing:
        print("Missing files:", *missing, sep="\n", file=sys.stderr)
        sys.exit(1)
    agg = aggregate_paths(paths)
    sys.stdout.write(format_report(agg))


if __name__ == "__main__":
    main()
