#!/usr/bin/env python3
"""Greedy MiniBG DQN vs heuristic bots (same protocol as heuristic tournament: seat swap)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.envs  # noqa: F401

from src.agents.dqn.agent import DQNAgent
from src.envs.minibg.env import MiniBGEnv
from src.envs.minibg.heuristic_bots.bots import HeuristicBot, default_bot_constructors
from src.envs.minibg.heuristic_bots.common import legal_env_indices
from src.envs.minibg.heuristic_bots.tournament import make_bot


class DQNPolicyBot(HeuristicBot):
    """Wraps a loaded DQN as a HeuristicBot (uses env self-centric obs)."""

    name = "dqn"
    order_style = "default"

    def __init__(self, agent: DQNAgent) -> None:
        super().__init__()
        self._agent = agent

    def choose_action(self, env: MiniBGEnv) -> int:
        obs = env._get_obs()
        return int(
            self._agent.act(
                obs, legal_mask=env.legal_actions_mask, deterministic=True
            )
        )


def play_game(
    bot0: HeuristicBot,
    bot1: HeuristicBot,
    seed: int,
    *,
    battle_damage_shaping: float,
    max_steps: int = 20_000,
) -> str:
    env = MiniBGEnv(seed=seed, battle_damage_shaping=battle_damage_shaping)
    rng = np.random.default_rng(seed + 17)
    steps = 0
    while not env.done and steps < max_steps:
        idx = env.current_player()
        bot = bot0 if idx == 0 else bot1
        mask = env.legal_actions_mask
        action = bot.choose_action(env)
        if not bool(mask[action]):
            legal = legal_env_indices(mask)
            if not legal:
                break
            action = int(rng.choice(legal))
        env.step(action)
        steps += 1
    if not env.done:
        return "draw"
    w = env.winner
    if w == 1:
        return "bot0"
    if w == -1:
        return "bot1"
    return "draw"


def play_pair_dqn_vs_name(
    dqn: DQNPolicyBot,
    opp_name: str,
    games: int,
    base_seed: int,
    *,
    battle_damage_shaping: float,
) -> tuple[int, int, int]:
    """(dqn_wins, opp_wins, draws) over ``games`` seeds × 2 seatings (like tournament)."""
    d_w = o_w = dr = 0
    for g in range(games):
        s = base_seed + g * 1009 + 13
        opp_a = make_bot(opp_name, s + 2)
        r1 = play_game(dqn, opp_a, s, battle_damage_shaping=battle_damage_shaping)
        if r1 == "bot0":
            d_w += 1
        elif r1 == "bot1":
            o_w += 1
        else:
            dr += 1
        opp_b = make_bot(opp_name, s + 101)
        r2 = play_game(opp_b, dqn, s + 100, battle_damage_shaping=battle_damage_shaping)
        if r2 == "bot1":
            d_w += 1
        elif r2 == "bot0":
            o_w += 1
        else:
            dr += 1
    return d_w, o_w, dr


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--games",
        type=int,
        default=40,
        help="Seeds per opponent (2 games per seed: seat swap), like tournament --games",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--battle-damage-shaping",
        type=float,
        default=0.06,
        help="Must match training env if you used shaping.",
    )
    p.add_argument(
        "--opponents",
        type=str,
        default="",
        help="Comma-separated bot names (default: all heuristics except random)",
    )
    p.add_argument(
        "--include-random",
        action="store_true",
        help="Also evaluate vs random heuristic bot",
    )
    args = p.parse_args()

    ctors = default_bot_constructors()
    if args.opponents.strip():
        names = [x.strip() for x in args.opponents.split(",") if x.strip()]
    else:
        names = sorted(k for k in ctors.keys() if args.include_random or k != "random")
    for n in names:
        if n not in ctors:
            raise SystemExit(f"Unknown bot {n!r}. Known: {sorted(ctors)}")

    agent = DQNAgent.load(
        str(args.checkpoint), device=args.device, load_optimizer=False
    )
    agent.eval()
    agent.epsilon = 0.0
    dqn_bot = DQNPolicyBot(agent)

    n_played = 2 * args.games
    print(
        f"checkpoint={args.checkpoint}\n"
        f"games_per_opponent={args.games} seeds → {n_played} games each (seat swap)\n"
        f"battle_damage_shaping={args.battle_damage_shaping}\n"
    )
    for name in names:
        dw, ow, dr = play_pair_dqn_vs_name(
            dqn_bot,
            name,
            args.games,
            args.seed,
            battle_damage_shaping=args.battle_damage_shaping,
        )
        total = dw + ow + dr
        wr = dw / total if total else 0.0
        print(f"{name:22s}  DQN {dw:4d}  opp {ow:4d}  draw {dr:3d}  win_rate {wr:.3f}")


if __name__ == "__main__":
    main()
