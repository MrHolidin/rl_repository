"""Evaluate checkpoints against fixed opponents (random, heuristic) over training progress."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import pandas as pd

import re as _re

import numpy as np

from src.agents import HeuristicAgent, RandomAgent, SmartHeuristicAgent
from src.agents.othello import OthelloHeuristicAgent
from src.agents.dqn.agent import DQNAgent
from src.envs import Connect4Env, RewardConfig
from src.envs.othello import OthelloEnv
from src.envs.connect4 import Connect4Game
from src.search.connect4.heuristic_minimax import make_connect4_heuristic_minimax_policy
from src.search.connect4.minimax_env_adapter import Connect4MinimaxEnvAdapter
from src.utils.match import play_match, play_match_batched

if TYPE_CHECKING:
    from src.envs.base import TurnBasedEnv


def _step_from_filename(name: str) -> Optional[int]:
    """Extract step number from checkpoint filename (e.g. dqn_2000.pt -> 2000)."""
    m = re.search(r"_(\d+)\.pt$", name, re.IGNORECASE)
    return int(m.group(1)) if m else None


def find_checkpoints(
    checkpoints_dir: Path,
    *,
    prefix: Optional[str] = None,
    sort_by_step: bool = True,
) -> List[tuple[Path, int]]:
    """Find checkpoint files and extract step numbers."""
    results: List[tuple[Path, int]] = []
    for p in checkpoints_dir.iterdir():
        if not p.suffix.lower() == ".pt":
            continue
        step = _step_from_filename(p.name)
        if step is None:
            continue
        if prefix is not None and not p.stem.startswith(prefix + "_"):
            continue
        results.append((p, step))
    if sort_by_step:
        results.sort(key=lambda x: x[1])
    return results


def load_dqn_checkpoint(
    path: Path,
    *,
    device: Optional[str] = None,
    seed: int = 42,
) -> DQNAgent:
    """Load DQN agent from checkpoint (eval mode, epsilon=0)."""
    agent = DQNAgent.load(
        str(path),
        device=device,
        seed=seed,
        load_optimizer=False,
    )
    agent.eval()
    agent.epsilon = 0.0
    return agent


_VALID_OPPONENT_NAMES = frozenset({
    "random", "heuristic", "smart_heuristic", "othello_heuristic"
})

_MINIMAX_OPPONENT_PATTERN = _re.compile(r"^minimax_(\d+)$")

_OPPONENT_CLASSES = {
    "random": RandomAgent,
    "heuristic": HeuristicAgent,
    "smart_heuristic": SmartHeuristicAgent,
    "othello_heuristic": OthelloHeuristicAgent,
}


def _is_minimax_opponent(name: str) -> bool:
    return _MINIMAX_OPPONENT_PATTERN.match(name) is not None


def _minimax_depth(name: str) -> int:
    m = _MINIMAX_OPPONENT_PATTERN.match(name)
    return int(m.group(1)) if m else 0


def build_opponents_from_names(
    names: List[str],
    seed: int = 42,
    game_id: str = "connect4",
) -> Dict[str, "BaseAgent"]:
    """Build opponent agents from names (random, heuristic, smart_heuristic, othello_heuristic, minimax_N)."""
    from src.agents.base_agent import BaseAgent

    result: Dict[str, "BaseAgent"] = {}
    for i, name in enumerate(names):
        if _is_minimax_opponent(name):
            if game_id != "connect4":
                raise ValueError(f"minimax_N opponents only supported for connect4, got game_id={game_id}")
            depth = _minimax_depth(name)
            game = Connect4Game(rows=6, cols=7)
            policy = make_connect4_heuristic_minimax_policy(
                game, depth=depth, heuristic="trivial", rng=np.random.default_rng(seed + i + 1)
            )
            result[name] = Connect4MinimaxEnvAdapter(game=game, policy=policy, get_state=None)
        elif name in _OPPONENT_CLASSES:
            result[name] = _OPPONENT_CLASSES[name](seed=seed + i + 1)
        else:
            raise ValueError(
                f"Unknown opponent: {name}. Valid: {sorted(_VALID_OPPONENT_NAMES)} or minimax_N (e.g. minimax_2, minimax_4)"
            )
    return result


def _create_env(game_id: str, reward_config: Optional[RewardConfig] = None) -> "TurnBasedEnv":
    """Create environment by game_id."""
    if reward_config is None:
        reward_config = RewardConfig()
    if game_id == "othello":
        return OthelloEnv(size=8, reward_config=reward_config)
    else:
        return Connect4Env(rows=6, cols=7, reward_config=reward_config)


def eval_checkpoints_vs_opponents(
    checkpoint_paths: List[Path],
    *,
    opponents: Optional[Dict[str, "BaseAgent"]] = None,
    opponent_names: Optional[List[str]] = None,
    num_games: int = 100,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
    seed: int = 42,
    reward_config: Optional[RewardConfig] = None,
    start_policy: str = "random",
    game_id: str = "connect4",
    out_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Evaluate each checkpoint against each opponent.

    Args:
        checkpoint_paths: List of checkpoint file paths.
        opponents: Dict of {name: agent}. Ignored if opponent_names is set.
        opponent_names: List of opponent types: random, heuristic, smart_heuristic, othello_heuristic, minimax_N.
        num_games: Games per (checkpoint, opponent) pair.
        batch_size: Parallel envs for batched eval. None = use num_games (one batch).
        device: Device for DQN ('cuda' or 'cpu').
        seed: Random seed.
        reward_config: Environment reward config.
        start_policy: Who goes first: 'random', 'agent_first', or 'opponent_first'.
        game_id: Game identifier ('connect4' or 'othello').
        out_csv: If set, save DataFrame with exact wins/draws/losses to this path.

    Returns:
        DataFrame with columns: step, checkpoint, win_rate_*, draw_rate_*, loss_rate_*, wins_*, draws_*, losses_*, games_*.
    """
    from src.agents.base_agent import BaseAgent

    if batch_size is None:
        batch_size = num_games

    if reward_config is None:
        reward_config = RewardConfig()
    env = _create_env(game_id, reward_config)

    if opponent_names is not None:
        opponents = build_opponents_from_names(opponent_names, seed=seed, game_id=game_id)
    elif opponents is None:
        opponents = {
            "random": RandomAgent(seed=seed + 1),
            "heuristic": HeuristicAgent(seed=seed + 2),
        }

    rows: List[Dict] = []
    n_checkpoints = len(checkpoint_paths)
    for i, path in enumerate(checkpoint_paths, 1):
        step = _step_from_filename(path.name)
        if step is None:
            continue

        print(f"Evaluating checkpoint {i}/{n_checkpoints} (step {step})...", flush=True)
        agent = load_dqn_checkpoint(path, device=device, seed=seed)
        row: Dict = {"step": step, "checkpoint": path.stem}

        for opp_name, opponent in opponents.items():
            for a in (agent, opponent):
                if hasattr(a, "eval"):
                    a.eval()
                if hasattr(a, "epsilon"):
                    setattr(a, "epsilon", 0.0)

            sp = (start_policy or "random").strip().lower()
            randomize_first = sp == "random"
            agent_first_in_call = sp != "opponent_first"

            if agent_first_in_call:
                a1, a2 = agent, opponent
            else:
                a1, a2 = opponent, agent

            match_seed = seed + hash(path.stem) % (2**16) + hash(opp_name) % (2**16)
            if batch_size > 0:
                w1, draws, w2 = play_match_batched(
                    a1, a2,
                    num_games=num_games,
                    batch_size=min(batch_size, num_games),
                    seed=match_seed,
                    randomize_first_player=randomize_first,
                    reward_config=reward_config,
                    game_id=game_id,
                )
            else:
                w1, draws, w2 = play_match(
                    a1, a2,
                    num_games=num_games,
                    seed=match_seed,
                    randomize_first_player=randomize_first,
                    reward_config=reward_config,
                    env=env,
                )
            wins = w1 if agent_first_in_call else w2
            losses = w2 if agent_first_in_call else w1
            total = wins + draws + losses
            row[f"win_rate_{opp_name}"] = wins / total if total else 0.0
            row[f"draw_rate_{opp_name}"] = draws / total if total else 0.0
            row[f"loss_rate_{opp_name}"] = losses / total if total else 0.0
            row[f"wins_{opp_name}"] = wins
            row[f"draws_{opp_name}"] = draws
            row[f"losses_{opp_name}"] = losses
            row[f"games_{opp_name}"] = total

        rows.append(row)
        win_str = " | ".join(f"{k}: {row.get(k, 0):.2f}" for k in row if k.startswith("win_rate_"))
        print(f"  → {win_str}", flush=True)

    df = pd.DataFrame(rows)
    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"Saved eval results to {out_csv}", flush=True)
    return df


def eval_single_checkpoint_by_side(
    checkpoint_path: Path,
    *,
    opponent_name: str = "smart_heuristic",
    num_games_per_side: int = 200,
    device: Optional[str] = None,
    seed: int = 42,
    reward_config: Optional[RewardConfig] = None,
    game_id: str = "connect4",
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate one checkpoint vs one opponent, 200 games with DQN first and 200 with DQN second.
    Returns win/draw/loss rates per side.
    """
    if reward_config is None:
        reward_config = RewardConfig()
    env = _create_env(game_id, reward_config)
    opponents = build_opponents_from_names([opponent_name], seed=seed, game_id=game_id)
    opponent = opponents[opponent_name]
    agent = load_dqn_checkpoint(checkpoint_path, device=device, seed=seed)
    for a in (agent, opponent):
        if hasattr(a, "eval"):
            a.eval()
        if hasattr(a, "epsilon"):
            setattr(a, "epsilon", 0.0)

    # DQN first (agent1 = DQN)
    w1, d1, l1 = play_match(
        agent, opponent,
        num_games=num_games_per_side,
        seed=seed,
        randomize_first_player=False,
        reward_config=reward_config,
        env=env,
    )
    total1 = w1 + d1 + l1
    # DQN second (agent2 = DQN)
    l2, d2, w2 = play_match(
        opponent, agent,
        num_games=num_games_per_side,
        seed=seed + 10000,
        randomize_first_player=False,
        reward_config=reward_config,
        env=env,
    )
    total2 = l2 + d2 + w2

    return {
        "dqn_first": {
            "win_rate": w1 / total1 if total1 else 0.0,
            "draw_rate": d1 / total1 if total1 else 0.0,
            "loss_rate": l1 / total1 if total1 else 0.0,
            "wins": w1,
            "draws": d1,
            "losses": l1,
            "games": total1,
        },
        "dqn_second": {
            "win_rate": w2 / total2 if total2 else 0.0,
            "draw_rate": d2 / total2 if total2 else 0.0,
            "loss_rate": l2 / total2 if total2 else 0.0,
            "wins": w2,
            "draws": d2,
            "losses": l2,
            "games": total2,
        },
    }
