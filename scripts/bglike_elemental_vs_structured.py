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


def run_comparison(
    n_games: int,
    patch_dir: str,
    *,
    bot_a: str = "elemental",
    bot_b: str = "structured",
    seats_a: int = 2,
    quiet: bool = False,
) -> None:
    """Seats [0, seats_a) play ``bot_a``; the rest play ``bot_b``.

    Seat order alternates per game so seat bias (turn order, shop RNG) cancels.
    """
    a_placements: list[int] = []
    b_placements: list[int] = []

    for game_i in range(n_games):
        seed = game_i * 13 + 7
        # Rotate which seats belong to A so neither bot owns a fixed seat.
        offset = (game_i * seats_a) % NUM_PLAYERS
        a_seats = {(offset + k) % NUM_PLAYERS for k in range(seats_a)}
        bot_names = [bot_a if s in a_seats else bot_b for s in range(NUM_PLAYERS)]
        placements = run_lobby(bot_names, seed=seed, patch_dir=patch_dir)
        for s in range(NUM_PLAYERS):
            (a_placements if s in a_seats else b_placements).append(placements[s])

        if not quiet:
            avg_a = sum(a_placements) / len(a_placements)
            avg_b = sum(b_placements) / len(b_placements)
            print(
                f"Game {game_i+1:3d}/{n_games}  {bot_a}={[placements[s] for s in sorted(a_seats)]}  "
                f"{bot_a}_avg={avg_a:.2f}  {bot_b}_avg={avg_b:.2f}"
            )

    print()
    print("=" * 55)
    avg_a = sum(a_placements) / len(a_placements)
    avg_b = sum(b_placements) / len(b_placements)

    a_dist: dict[int, int] = defaultdict(int)
    b_dist: dict[int, int] = defaultdict(int)
    for p in a_placements:
        a_dist[p] += 1
    for p in b_placements:
        b_dist[p] += 1

    n_a = len(a_placements)
    n_b = len(b_placements)
    # Placement is a per-lobby ranking, so the two averages are coupled; the
    # standard error below treats seats as independent and is only a guide.
    var_a = sum((x - avg_a) ** 2 for x in a_placements) / max(1, n_a - 1)
    var_b = sum((x - avg_b) ** 2 for x in b_placements) / max(1, n_b - 1)
    se = (var_a / max(1, n_a) + var_b / max(1, n_b)) ** 0.5

    print(f"Games: {n_games}  |  {bot_a} seats: {seats_a}, {bot_b} seats: {NUM_PLAYERS - seats_a}")
    print(
        f"{bot_a}_avg={avg_a:.3f}  {bot_b}_avg={avg_b:.3f}  "
        f"diff={avg_a - avg_b:+.3f} (+-{1.96 * se:.3f})  (lower = better, equal skill = 4.5)"
    )
    print()
    for label, dist, n in ((bot_a, a_dist, n_a), (bot_b, b_dist, n_b)):
        print(f"{label} placement distribution  (n={n}):")
        for place in range(1, 9):
            cnt = dist.get(place, 0)
            bar = "#" * (cnt * 30 // max(n, 1))
            pct = cnt / max(n, 1) * 100
            print(f"  {place}: {bar:<30s} {cnt:3d}  ({pct:5.1f}%)")
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--patch", default=DEFAULT_PATCH)
    ap.add_argument("--bot-a", default="elemental")
    ap.add_argument("--bot-b", default="structured")
    ap.add_argument("--seats-a", type=int, default=2)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    print(f"Running {args.games} games  patch={args.patch}  {args.bot_a} x{args.seats_a} vs {args.bot_b}")
    print()
    run_comparison(
        args.games,
        args.patch,
        bot_a=args.bot_a,
        bot_b=args.bot_b,
        seats_a=args.seats_a,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
