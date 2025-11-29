"""Helpers to run evaluation games between agents."""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.agents.base_agent import BaseAgent
from src.envs.base import TurnBasedEnv
from src.training.opponent_sampler import OpponentSampler
from src.training.random_opening import RandomOpeningConfig, maybe_apply_random_opening
from src.training.trainer import StartPolicy


def play_single_game(
    env: TurnBasedEnv,
    agent: BaseAgent,
    opponent: BaseAgent,
    *,
    start_policy: StartPolicy = StartPolicy.RANDOM,
    random_opening_config: Optional[RandomOpeningConfig] = None,
    deterministic_agent: bool = True,
    deterministic_opponent: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Play a single self-contained game between ``agent`` and ``opponent`` on ``env``.

    Args:
        env: Environment instance (will be reset inside the function).
        agent: Main agent under evaluation.
        opponent: Opponent agent (can be heuristic, frozen policy, etc.).
        start_policy: Who plays first (agent, opponent, or random each game).
        random_opening_config: Optional random opening configuration applied before the game.
        deterministic_agent: Whether to call ``agent.act(..., deterministic=True)``.
        deterministic_opponent: Same flag for the opponent.
        seed: Optional per-game seed for reproducibility.

    Returns:
        Dictionary with keys ``winner`` (1/-1/0), ``steps``, ``agent_token`` and raw ``info`` from env.
    """
    rng = random.Random(seed) if seed is not None else None
    obs = env.reset()

    if random_opening_config is not None:
        opener_rng = rng if rng is not None else random
        obs, _, prologue_done = maybe_apply_random_opening(
            env=env,
            initial_obs=obs,
            config=random_opening_config,
            rng=opener_rng,
        )
        if prologue_done:
            obs = env.reset()

    agent_token = _resolve_agent_token(start_policy, rng)
    steps = 0
    done = False

    while not done:
        current_token = _current_player_token(env)
        is_agent_turn = current_token == agent_token
        actor = agent if is_agent_turn else opponent
        deterministic = deterministic_agent if is_agent_turn else deterministic_opponent
        legal_mask = getattr(env, "legal_actions_mask", None)
        legal_actions = _legal_actions_list(env, legal_mask)

        action = _choose_action(actor, obs, legal_mask, legal_actions, deterministic)
        step = env.step(action)
        steps += 1

        if step.done:
            winner = step.info.get("winner")
            reward = _result_to_reward(winner, agent_token)
            return {
                "winner": winner,
                "steps": steps,
                "agent_token": agent_token,
                "reward": reward,
                "info": step.info,
            }

        obs = step.obs

    raise RuntimeError("Game loop exited without reaching a terminal state.")


def play_series_vs_sampler(
    env_factory: Callable[[], TurnBasedEnv],
    agent: BaseAgent,
    opponent_sampler: OpponentSampler,
    *,
    num_games: int,
    start_policy: StartPolicy = StartPolicy.RANDOM,
    random_opening_config: Optional[RandomOpeningConfig] = None,
    deterministic_agent: bool = True,
    deterministic_opponent: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Play a series of games against opponents drawn from ``opponent_sampler``.

    Args:
        env_factory: Callable returning a fresh environment per game.
        agent: Agent under evaluation.
        opponent_sampler: Sampler that provides opponents (uses prepare/sample/on_episode_end).
        num_games: Number of games to play.
        start_policy: Who goes first (agent, opponent, or random).
        random_opening_config: Optional random opening applied before each game.
        deterministic_agent: Whether the main agent acts deterministically.
        deterministic_opponent: Whether sampled opponents act deterministically.
        seed: Optional seed controlling start policy randomness and openings.

    Returns:
        Summary dictionary with fields ``wins``, ``losses``, ``draws`` and per-game ``history``.
    """
    rng = random.Random(seed) if seed is not None else None
    wins = losses = draws = 0
    history: List[Dict[str, Any]] = []

    for game_idx in range(num_games):
        env = env_factory()
        opponent_sampler.prepare(game_idx)
        opponent = opponent_sampler.sample()
        if hasattr(opponent, "eval"):
            opponent.eval()
        if hasattr(opponent, "epsilon"):
            setattr(opponent, "epsilon", 0.0)

        per_game_seed = (rng.randrange(0, 2**32 - 1) if rng is not None else None)
        result = play_single_game(
            env,
            agent,
            opponent,
            start_policy=start_policy,
            random_opening_config=random_opening_config,
            deterministic_agent=deterministic_agent,
            deterministic_opponent=deterministic_opponent,
            seed=per_game_seed,
        )
        winner = result["winner"]
        if winner == result["agent_token"]:
            wins += 1
        elif winner == -result["agent_token"]:
            losses += 1
        else:
            draws += 1

        history.append(result)
        episode_info = {
            "reward": result["reward"],
            "length": result["steps"],
            "info": {"winner": winner},
        }
        opponent_sampler.on_episode_end(game_idx, episode_info)

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "history": history,
    }


def _resolve_agent_token(start_policy: StartPolicy, rng: Optional[random.Random]) -> int:
    if start_policy == StartPolicy.AGENT_FIRST:
        return 1
    if start_policy == StartPolicy.OPPONENT_FIRST:
        return -1
    coin = rng.random() if rng is not None else random.random()
    return 1 if coin < 0.5 else -1


def _current_player_token(env: TurnBasedEnv) -> int:
    if hasattr(env, "current_player_token"):
        return int(getattr(env, "current_player_token"))
    if hasattr(env, "current_player"):
        current = env.current_player()
        return 1 if current == 0 else -1
    raise AttributeError("Environment must expose current_player_token or current_player().")


def _choose_action(
    agent: BaseAgent,
    obs: Any,
    legal_mask: Optional[Any],
    legal_actions: Optional[Sequence[int]],
    deterministic: bool,
) -> int:
    if hasattr(agent, "act"):
        return agent.act(obs, legal_mask=legal_mask, deterministic=deterministic)
    if hasattr(agent, "select_action"):
        if legal_actions is None:
            raise ValueError("select_action requires explicit legal_actions list.")
        return agent.select_action(obs, list(legal_actions))
    raise AttributeError("Agent must implement act() or select_action().")


def _result_to_reward(winner: Optional[int], agent_token: int) -> int:
    if winner is None or winner == 0:
        return 0
    return 1 if winner == agent_token else -1


def _legal_actions_list(env: TurnBasedEnv, legal_mask: Optional[Any]) -> Optional[Sequence[int]]:
    if legal_mask is not None:
        return [idx for idx, allowed in enumerate(legal_mask) if allowed]
    if hasattr(env, "get_legal_actions"):
        actions = env.get_legal_actions()
        if actions is not None:
            return list(actions)
    return None

