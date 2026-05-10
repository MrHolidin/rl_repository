#!/usr/bin/env python3
"""Short MiniBG DQN train vs random, then greedy eval over 100 games (seat-swapped)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.envs  # noqa: F401 — register minibg / games

from src.agents.dqn.agent import DQNAgent
from src.agents.random_agent import RandomAgent
from src.registry import make_game
from src.training.control_flow import active_role
from src.training.run import run


def _latest_checkpoint(checkpoint_dir: Path) -> Path:
    paths = sorted(checkpoint_dir.glob("minibg_dqn_*.pt"), key=lambda p: p.stat().st_mtime)
    if not paths:
        raise FileNotFoundError(f"No minibg_dqn_*.pt under {checkpoint_dir}")
    return paths[-1]


def _wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for Binomial(wins, n)."""
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, center - margin), min(1.0, center + margin))


def play_series(
    *,
    n_games: int,
    battle_damage_shaping: float,
    rng: np.random.Generator,
    build_policies: Callable[
        [np.random.Generator], Tuple[Callable[..., int], Callable[..., int]]
    ],
    label: str,
) -> dict:
    wins = losses = draws = 0
    none_w = 0
    wins_as_p0 = wins_as_p1 = 0
    n_as_p0 = n_as_p1 = 0
    for g in range(n_games):
        agent_token = 1 if (g % 2 == 0) else -1
        env = make_game(
            "minibg",
            seed=int(rng.integers(0, 2**31 - 1)),
            battle_damage_shaping=battle_damage_shaping,
        )
        obs = env.reset()
        play_agent, play_opp = build_policies(rng)
        while not env.done:
            role = active_role(env, agent_token)
            mask = env.legal_actions_mask
            if role == "agent":
                a = play_agent(obs, mask)
            else:
                a = play_opp(obs, mask)
            obs = env.step(a).obs

        w = env.winner
        if w is None:
            none_w += 1
            losses += 1
            continue
        if w == 0:
            draws += 1
        elif w == agent_token:
            wins += 1
        else:
            losses += 1
        if agent_token == 1:
            n_as_p0 += 1
            if w == 1:
                wins_as_p0 += 1
        else:
            n_as_p1 += 1
            if w == -1:
                wins_as_p1 += 1

    lo, hi = _wilson_ci(wins, n_games)
    out = {
        "label": label,
        "n": n_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "none_winner": none_w,
        "win_rate": wins / n_games,
        "wilson_95": (lo, hi),
        "wins_as_p0": wins_as_p0,
        "n_as_p0": n_as_p0,
        "wins_as_p1": wins_as_p1,
        "n_as_p1": n_as_p1,
    }
    return out


def _print_series(stats: dict) -> None:
    lo, hi = stats["wilson_95"]
    p0 = stats["wins_as_p0"] / max(1, stats["n_as_p0"])
    p1 = stats["wins_as_p1"] / max(1, stats["n_as_p1"])
    print(
        f"{stats['label']}: n={stats['n']} W/L/D={stats['wins']}/{stats['losses']}/{stats['draws']} "
        f"none_winner={stats['none_winner']} win_rate={stats['win_rate']:.4f} "
        f"Wilson95%=[{lo:.3f},{hi:.3f}]"
    )
    print(
        f"  as P0 (+1 token): {stats['wins_as_p0']}/{stats['n_as_p0']} ({p0:.4f}) | "
        f"as P1 (-1): {stats['wins_as_p1']}/{stats['n_as_p1']} ({p1:.4f})"
    )


def eval_vs_random(
    checkpoint: Path,
    *,
    n_games: int,
    battle_damage_shaping: float,
    seed: int,
    device: str,
    baseline_random: bool,
) -> None:
    agent = DQNAgent.load(str(checkpoint), device=device, load_optimizer=False)
    agent.eval()
    agent.epsilon = 0.0

    rng = np.random.default_rng(seed + 555)

    if baseline_random:

        def build_rr(r: np.random.Generator):
            ra = RandomAgent(seed=int(r.integers(0, 2**31 - 1)))
            rb = RandomAgent(seed=int(r.integers(0, 2**31 - 1)))
            return (
                lambda obs, m, _ra=ra: _ra.act(obs, legal_mask=m, deterministic=False),
                lambda obs, m, _rb=rb: _rb.act(obs, legal_mask=m, deterministic=False),
            )

        br = play_series(
            n_games=n_games,
            battle_damage_shaping=battle_damage_shaping,
            rng=np.random.default_rng(seed + 777),
            build_policies=build_rr,
            label="baseline Random vs Random (same harness)",
        )
        _print_series(br)

    def build_dqn_vs_r(r: np.random.Generator):
        opp = RandomAgent(seed=int(r.integers(0, 2**31 - 1)))
        return (
            lambda obs, m: agent.act(obs, legal_mask=m, deterministic=True),
            lambda obs, m, _o=opp: _o.act(obs, legal_mask=m, deterministic=False),
        )

    st = play_series(
        n_games=n_games,
        battle_damage_shaping=battle_damage_shaping,
        rng=rng,
        build_policies=build_dqn_vs_r,
        label=f"DQN greedy vs Random checkpoint={checkpoint.name}",
    )
    _print_series(st)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/minibg/dqn_vs_random_shaping.yaml",
    )
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--n-games", type=int, default=100)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--baseline-random",
        action="store_true",
        help="Also run Random vs Random with the same seat alternation (sanity check ~50%% wins).",
    )
    args = p.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_train:
        run(args.config, args.run_dir)

    ckpt = args.checkpoint or _latest_checkpoint(args.run_dir / "checkpoints")
    shaping = 0.06
    try:
        import yaml

        data = yaml.safe_load(args.config.read_text())
        shaping = float((data.get("game") or {}).get("params") or {}).get(
            "battle_damage_shaping", shaping
        )
    except Exception:
        pass

    seed = 7
    try:
        import yaml

        data = yaml.safe_load(args.config.read_text())
        if data.get("seed") is not None:
            seed = int(data["seed"])
    except Exception:
        pass

    eval_vs_random(
        ckpt,
        n_games=args.n_games,
        battle_damage_shaping=shaping,
        seed=seed,
        device=args.device,
        baseline_random=args.baseline_random,
    )


if __name__ == "__main__":
    main()
