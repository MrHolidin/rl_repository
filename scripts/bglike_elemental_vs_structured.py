"""Compare ElementalHeuristicBot vs StructuredHeuristicBot in 8-player lobby.

Usage:
    python3 scripts/bglike_elemental_vs_structured.py [--games N] [--patch PATH]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from typing import Optional

sys.path.insert(0, ".")

import numpy as np

from src.agents.random_agent import RandomAgent
from src.envs.bglike.heuristic_bots.bots import make_bot
from src.envs.bglike.lobby_env import BGLobbyEnv
from src.envs.bglike.seat_config import SeatConfig, SeatKind

NUM_PLAYERS = 8
DEFAULT_PATCH = "data/bgcore/19_6_0_74257"


class DirectBotView:
    """Minimal env view that wraps BGLobbyEnv for heuristic bots."""

    def __init__(self, lobby: BGLobbyEnv, seat: int) -> None:
        self._lobby = lobby
        self._seat = seat
        self._mask_override: Optional[np.ndarray] = None

    @property
    def state(self):
        return self._lobby.state

    @property
    def _game(self):
        return self._lobby._game

    @property
    def patch(self):
        return self._lobby._game._patch

    @property
    def rl_pending(self):
        return self._lobby.rl_pending_for_seat(self._seat)

    def current_player(self) -> int:
        return self._seat

    def set_mask_override(self, mask: Optional[np.ndarray]) -> None:
        self._mask_override = None if mask is None else np.asarray(mask, dtype=bool)

    @property
    def legal_actions_mask(self) -> np.ndarray:
        if self._mask_override is not None:
            return self._mask_override
        return self._lobby.legal_mask_for_seat(self._seat)


def run_lobby(bot_names: list[str], *, seed: int, patch_dir: str) -> dict[int, int]:
    """Run one full 8p lobby with the given bot per seat. Returns {seat: placement}."""
    dummy = RandomAgent(seed=seed)
    env = BGLobbyEnv(
        [SeatConfig(SeatKind.LEARNED, dummy)] + [SeatConfig(SeatKind.RANDOM)] * 7,
        learned_seats=[0],
        seed=seed,
        patch_dir=patch_dir,
    )
    bots = [make_bot(name, seed=seat + seed * 100) for seat, name in enumerate(bot_names)]
    views = [DirectBotView(env, seat) for seat in range(NUM_PLAYERS)]

    env.reset()
    steps = 0
    while not env.lobby_done and steps < 20_000:
        steps += 1
        cur = env.current_seat()
        mask = env.legal_mask_for_seat(cur)
        view = views[cur]
        view.set_mask_override(mask)
        try:
            action = bots[cur].choose_action(view)
        finally:
            view.set_mask_override(None)
        env.step_action(cur, action)

    return env.finalize_placements()


def run_comparison(n_games: int, patch_dir: str) -> None:
    # Seats 0,1 = elemental; seats 2-7 = structured
    elemental_placements: list[int] = []
    structured_placements: list[int] = []

    for game_i in range(n_games):
        seed = game_i * 13 + 7
        bot_names = ["elemental", "elemental"] + ["structured"] * 6
        placements = run_lobby(bot_names, seed=seed, patch_dir=patch_dir)
        for s in range(2):
            elemental_placements.append(placements[s])
        for s in range(2, NUM_PLAYERS):
            structured_placements.append(placements[s])

        e_places = [placements[0], placements[1]]
        avg_e = sum(elemental_placements) / len(elemental_placements)
        avg_s = sum(structured_placements) / len(structured_placements)
        print(f"Game {game_i+1:3d}/{n_games}  elemental={e_places}  "
              f"elemental_avg={avg_e:.2f}  structured_avg={avg_s:.2f}")

    print()
    print("=" * 55)
    n = n_games
    avg_e = sum(elemental_placements) / len(elemental_placements)
    avg_s = sum(structured_placements) / len(structured_placements)

    e_dist: dict[int, int] = defaultdict(int)
    s_dist: dict[int, int] = defaultdict(int)
    for p in elemental_placements:
        e_dist[p] += 1
    for p in structured_placements:
        s_dist[p] += 1

    n_e = len(elemental_placements)
    n_s = len(structured_placements)
    print(f"Games: {n}  |  elemental seats: 2, structured seats: 6")
    print(f"elemental_avg={avg_e:.3f}  structured_avg={avg_s:.3f}  (lower = better, equal-skill baseline = 4.5)")
    print()
    print(f"Elemental placement distribution  (n={n_e}):")
    for place in range(1, 9):
        cnt = e_dist.get(place, 0)
        bar = "#" * (cnt * 30 // max(n_e, 1))
        pct = cnt / n_e * 100
        print(f"  {place}: {bar:<30s} {cnt:3d}  ({pct:5.1f}%)")
    print()
    print(f"Structured placement distribution  (n={n_s}):")
    for place in range(1, 9):
        cnt = s_dist.get(place, 0)
        pct = cnt / n_s * 100
        print(f"  {place}: {'#' * (cnt * 30 // n_s):<30s} {cnt:3d}  ({pct:5.1f}%)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--patch", default=DEFAULT_PATCH)
    args = ap.parse_args()
    print(f"Running {args.games} games  patch={args.patch}")
    print()
    run_comparison(args.games, args.patch)


if __name__ == "__main__":
    main()
