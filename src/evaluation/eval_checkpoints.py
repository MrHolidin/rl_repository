"""Evaluate checkpoints against fixed opponents (random, heuristic) over training progress."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import pandas as pd

import re as _re

import numpy as np

from src.agents import HeuristicAgent, RandomAgent, SmartHeuristicAgent
from src.utils import freeze_agent
from src.agents.othello import OthelloHeuristicAgent
from src.agents.dqn.agent import DQNAgent
from src.envs import Connect4Env, RewardConfig
from src.envs.othello import OthelloEnv
from src.envs.connect4 import Connect4Game
from src.search.connect4.heuristic_minimax import make_connect4_heuristic_minimax_policy
from src.search.connect4.minimax_env_adapter import Connect4MinimaxEnvAdapter
from src.training.trainer import StartPolicy
from src.envs.minibg.replay_render import render_jsonl_file
from src.utils.match import (
    play_match,
    play_match_batched,
    play_single_game,
    resolve_opening_agent_token,
)

if TYPE_CHECKING:
    from src.envs.base import TurnBasedEnv


def _step_from_filename(name: str) -> Optional[int]:
    """Extract step number from checkpoint filename (e.g. dqn_2000.pt -> 2000)."""
    m = re.search(r"_(\d+)\.pt$", name, re.IGNORECASE)
    if m:
        return int(m.group(1))
    if re.search(r"_final\.pt$", name, re.IGNORECASE):
        return 2**31 - 1
    return None


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
    freeze_agent(agent)
    return agent


def load_training_agent_checkpoint(
    path: Path,
    *,
    device: Optional[str] = None,
    seed: int = 42,
) -> Any:
    """Load DQN or PPO checkpoint (eval exploit mode)."""
    import torch

    from src.agents.ppo_agent import PPOAgent

    path_str = str(path)
    map_location = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path_str, map_location=map_location)
    if ckpt.get("agent_kind") == "ppo_minibg_structured":
        from src.agents.ppo_structured_minibg_agent import MiniBGPPOStructuredAgent

        agent = MiniBGPPOStructuredAgent.load(path_str, device=device, seed=seed)
        freeze_agent(agent)
        return agent
    if "policy_state_dict" in ckpt:
        agent = PPOAgent.load(path_str, device=device, seed=seed)
        freeze_agent(agent)
        return agent
    if "q_network_state_dict" in ckpt:
        return load_dqn_checkpoint(Path(path_str), device=device, seed=seed)
    raise ValueError(
        f"Unknown checkpoint layout in {path_str!r}: "
        "expected policy_state_dict (PPO) or q_network_state_dict (DQN)."
    )


def evaluate_agent_vs_opponents_metrics(
    agent: Any,
    *,
    opponents: Dict[str, Any],
    num_games: int,
    batch_size: Optional[int],
    game_id: str = "connect4",
    seed: int = 42,
    reward_config: Optional[RewardConfig] = None,
    start_policy: str = "random",
    minibg_params: Optional[Dict[str, Any]] = None,
    replay_dir: Optional[Path] = None,
    match_identity: str = "checkpoint",
    deterministic: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate one trained agent vs a fixed opponent dict (fair match / batched replay path).
    Mirrors ``eval_checkpoints_vs_opponents`` per-checkpoint opponent loop.
    """
    from src.registry import make_game

    if reward_config is None:
        reward_config = RewardConfig()
    if batch_size is None:
        batch_size = num_games

    gid = str(game_id).strip().lower()
    use_replay = replay_dir is not None and gid == "minibg"
    if replay_dir is not None and not use_replay:
        raise ValueError("replay_dir is only supported when game_id=minibg")

    env: Optional["TurnBasedEnv"] = None
    if not use_replay:
        if gid == "minibg" and minibg_params:
            import src.envs  # noqa: F401

            env = make_game("minibg", reward_config=reward_config, **minibg_params)
        else:
            env = _create_env(game_id, reward_config)

    metrics: Dict[str, Any] = {}

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

        match_seed = seed + hash(match_identity) % (2**16) + hash(opp_name) % (2**16)

        if use_replay:
            replay_dir.mkdir(parents=True, exist_ok=True)
            w1 = draws = w2 = 0
            mg_base = dict(minibg_params or {})
            from src.agents.ppo_structured_minibg_agent import MiniBGPPOStructuredAgent

            learned_kind = (
                "ppo_minibg_structured"
                if isinstance(agent, MiniBGPPOStructuredAgent)
                else type(agent).__name__
            )
            for g in range(num_games):
                rpath = replay_dir / f"{match_identity}__{opp_name}__{g:04d}.jsonl"
                game_seed = (match_seed + g) if match_seed is not None else None
                inner_start = StartPolicy.RANDOM if randomize_first else StartPolicy.AGENT_FIRST
                agent_token = resolve_opening_agent_token(inner_start, seed=game_seed)
                learned_player_index = 0 if agent_token == 1 else 1
                scripted_player_index = 1 - learned_player_index
                meta = {
                    "checkpoint": match_identity,
                    "opponent": opp_name,
                    "game_index": g,
                    "match_seed": match_seed,
                    "game_seed": game_seed,
                    "opening_policy": inner_start.name,
                    "agent_token": agent_token,
                    "learned_player_index": learned_player_index,
                    "scripted_player_index": scripted_player_index,
                    "learned_agent_kind": learned_kind,
                    "scripted_opponent": opp_name,
                    "roles_note": (
                        f"P{learned_player_index}=learned({learned_kind}); "
                        f"P{scripted_player_index}=scripted({opp_name})"
                    ),
                }
                env_g = make_game(
                    "minibg",
                    reward_config=reward_config,
                    replay_path=rpath,
                    replay_meta=meta,
                    **mg_base,
                )
                for a in (agent, opponent):
                    if hasattr(a, "set_env"):
                        a.set_env(env_g)
                a1x, a2x = (agent, opponent) if agent_first_in_call else (opponent, agent)
                result = play_single_game(
                    env_g,
                    a1x,
                    a2x,
                    start_policy=inner_start,
                    random_opening_config=None,
                    deterministic_agent=deterministic,
                    deterministic_opponent=True,
                    seed=game_seed,
                )
                if hasattr(env_g, "close_replay"):
                    env_g.close_replay()
                rpath.with_suffix(".txt").write_text(render_jsonl_file(rpath), encoding="utf-8")
                winner = result["winner"]
                if winner == 0:
                    draws += 1
                elif winner == result["agent_token"]:
                    w1 += 1
                else:
                    w2 += 1
        elif batch_size > 0:
            env_factory = None
            if gid == "minibg" and minibg_params:
                mg = dict(minibg_params)

                def _minibg_env_factory(rc: RewardConfig) -> "TurnBasedEnv":
                    return make_game("minibg", reward_config=rc, **mg)

                env_factory = _minibg_env_factory
            w1, draws, w2 = play_match_batched(
                a1,
                a2,
                num_games=num_games,
                batch_size=min(batch_size, num_games),
                seed=match_seed,
                randomize_first_player=randomize_first,
                reward_config=reward_config,
                game_id=game_id,
                env_factory=env_factory,
                deterministic=deterministic,
            )
        else:
            w1, draws, w2 = play_match(
                a1,
                a2,
                num_games=num_games,
                seed=match_seed,
                randomize_first_player=randomize_first,
                reward_config=reward_config,
                env=env,
                game_id=game_id,
            )
        wins = w1 if agent_first_in_call else w2
        losses = w2 if agent_first_in_call else w1
        total = wins + draws + losses
        metrics[f"win_rate_{opp_name}"] = wins / total if total else 0.0
        metrics[f"draw_rate_{opp_name}"] = draws / total if total else 0.0
        metrics[f"loss_rate_{opp_name}"] = losses / total if total else 0.0
        metrics[f"wins_{opp_name}"] = wins
        metrics[f"draws_{opp_name}"] = draws
        metrics[f"losses_{opp_name}"] = losses
        metrics[f"games_{opp_name}"] = total

    return metrics


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
    """Build opponent agents from names (Connect4/Othello or MiniBG heuristic keys)."""
    from src.agents.base_agent import BaseAgent

    result: Dict[str, "BaseAgent"] = {}
    gid = (game_id or "connect4").strip().lower()

    if gid == "minibg":
        from src.envs.minibg.heuristic_bots.agent_adapter import MiniBGHeuristicAgent
        from src.envs.minibg.heuristic_bots.tournament import make_bot
        from src.envs.minibg.heuristic_bots.bots import default_bot_constructors

        valid = frozenset(default_bot_constructors().keys())
        for i, name in enumerate(names):
            if name not in valid:
                raise ValueError(
                    f"Unknown minibg opponent: {name!r}. Valid: {sorted(valid)}"
                )
            result[name] = MiniBGHeuristicAgent(make_bot(name, seed + i + 1))
        return result

    for i, name in enumerate(names):
        if _is_minimax_opponent(name):
            if gid != "connect4":
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
    gid = (game_id or "connect4").strip().lower()
    if gid == "othello":
        return OthelloEnv(size=8, reward_config=reward_config)
    if gid == "minibg":
        import src.envs  # noqa: F401
        from src.registry import make_game

        return make_game("minibg", reward_config=reward_config)
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
    replay_dir: Optional[Path] = None,
    minibg_params: Optional[Dict[str, Any]] = None,
    deterministic: bool = True,
) -> pd.DataFrame:
    """
    Evaluate each checkpoint against each opponent.

    Args:
        checkpoint_paths: List of checkpoint file paths.
        opponents: Dict of {name: agent}. Ignored if opponent_names is set.
        opponent_names: List of opponent types: random, heuristic, smart_heuristic, othello_heuristic, minimax_N.
        num_games: Games per (checkpoint, opponent) pair.
        batch_size: Parallel envs for batched eval. None = use num_games (one batch).
        device: Device for DQN/PPO tensors (``cuda`` or ``cpu``). ``None`` = CUDA if available.
            ``src.cli.eval_progress`` passes ``cpu`` for ``game_id=minibg`` when ``--device`` is omitted.
        seed: Random seed.
        reward_config: Environment reward config.
        start_policy: Who goes first: 'random', 'agent_first', or 'opponent_first'.
        game_id: ``connect4``, ``othello``, or ``minibg`` (MiniBG opponents: heuristic bot names).
        out_csv: If set, save DataFrame with exact wins/draws/losses to this path.
        replay_dir: If set with ``game_id=minibg``, write one JSONL replay per game under this directory
            plus a sibling ``.txt`` human-readable render (sequential eval; batched eval is not used).
            Typical size is well under 1 MB per dozen short games.
        minibg_params: Extra kwargs for ``make_game("minibg", ...)`` (e.g. ``battle_damage_shaping``).
        deterministic: Policy determinism during batched / replay branches (passed to matchup).

    Returns:
        DataFrame with columns: step, checkpoint, win_rate_*, draw_rate_*, loss_rate_*, wins_*, draws_*, losses_*, games_*.
    """
    if batch_size is None:
        batch_size = num_games

    if reward_config is None:
        reward_config = RewardConfig()

    gid = str(game_id).strip().lower()
    if replay_dir is not None and gid != "minibg":
        raise ValueError("replay_dir is only supported when game_id=minibg")

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
        agent = load_training_agent_checkpoint(path, device=device, seed=seed)
        row_metrics = evaluate_agent_vs_opponents_metrics(
            agent,
            opponents=opponents,
            num_games=num_games,
            batch_size=batch_size,
            game_id=game_id,
            seed=seed,
            reward_config=reward_config,
            start_policy=start_policy,
            minibg_params=minibg_params,
            replay_dir=replay_dir,
            match_identity=path.stem,
            deterministic=deterministic,
        )
        row: Dict[str, Any] = {"step": step, "checkpoint": path.stem, **row_metrics}

        rows.append(row)
        win_str = " | ".join(
            f"{k}: {row.get(k, 0):.2f}" for k in row if k.startswith("win_rate_")
        )
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
    Evaluate one checkpoint vs one opponent, 200 games with the learned agent first
    and 200 with the learned agent second (Connect4-style fixed side).
    Returns win/draw/loss rates per side.
    """
    if reward_config is None:
        reward_config = RewardConfig()
    env = _create_env(game_id, reward_config)
    opponents = build_opponents_from_names([opponent_name], seed=seed, game_id=game_id)
    opponent = opponents[opponent_name]
    agent = load_training_agent_checkpoint(checkpoint_path, device=device, seed=seed)
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
