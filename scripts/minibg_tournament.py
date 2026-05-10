#!/usr/bin/env python3
"""Run Mini BG heuristic tournament from repo root: python scripts/minibg_tournament.py"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.minibg.heuristic_bots.tournament import main

if __name__ == "__main__":
    main()
