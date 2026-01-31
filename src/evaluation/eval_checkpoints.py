"""Evaluate checkpoints against fixed opponents (random, heuristic) over training progress."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.agents import HeuristicAgent, RandomAgent, SmartHeuristicAgent
from src.agents.dqn_agent import DQNAgent
from src.envs import Connect4Env, RewardConfig
from src.features.action_space import DiscreteActionSpace
from src.features.observation_builder import BoardChannels
from src.utils.match import play_match


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


_VALID_OPPONENT_NAMES = frozenset({"random", "heuristic", "smart_heuristic"})


def build_opponents_from_names(
    names: List[str],
    seed: int = 42,
) -> Dict[str, "BaseAgent"]:
    """Build opponent agents from names (random, heuristic, smart_heuristic)."""
    from src.agents.base_agent import BaseAgent

    bad = [n for n in names if n not in _VALID_OPPONENT_NAMES]
    if bad:
        raise ValueError(f"Unknown opponent names: {bad}. Valid: {sorted(_VALID_OPPONENT_NAMES)}")
    agents = {
        "random": RandomAgent,
        "heuristic": HeuristicAgent,
        "smart_heuristic": SmartHeuristicAgent,
    }
    return {name: agents[name](seed=seed + (i + 1)) for i, name in enumerate(names)}


def eval_checkpoints_vs_opponents(
    checkpoint_paths: List[Path],
    *,
    opponents: Optional[Dict[str, "BaseAgent"]] = None,
    opponent_names: Optional[List[str]] = None,
    num_games: int = 100,
    device: Optional[str] = None,
    seed: int = 42,
    reward_config: Optional[RewardConfig] = None,
    start_policy: str = "random",
) -> pd.DataFrame:
    """
    Evaluate each checkpoint against each opponent.

    Args:
        checkpoint_paths: List of checkpoint file paths.
        opponents: Dict of {name: agent}. Ignored if opponent_names is set.
        opponent_names: List of opponent types: random, heuristic, smart_heuristic. Builds opponents if given.
        num_games: Games per (checkpoint, opponent) pair.
        device: Device for DQN ('cuda' or 'cpu').
        seed: Random seed.
        reward_config: Environment reward config.
        start_policy: Who goes first: 'random', 'agent_first', or 'opponent_first'.

    Returns:
        DataFrame with columns: step, win_rate_<opponent>, draw_rate_<opponent>, ...
    """
    from src.agents.base_agent import BaseAgent

    if reward_config is None:
        reward_config = RewardConfig()
    env = Connect4Env(rows=6, cols=7, reward_config=reward_config)

    if opponent_names is not None:
        opponents = build_opponents_from_names(opponent_names, seed=seed)
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

            w1, draws, w2 = play_match(
                a1,
                a2,
                num_games=num_games,
                seed=seed + hash(path.stem) % (2**16) + hash(opp_name) % (2**16),
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

        rows.append(row)
        win_str = " | ".join(f"{k}: {row.get(k, 0):.2f}" for k in row if k.startswith("win_rate_"))
        print(f"  → {win_str}", flush=True)

    return pd.DataFrame(rows)
