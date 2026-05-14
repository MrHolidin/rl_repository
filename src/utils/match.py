"""Utilities for playing matches between agents."""

import random
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from src.agents.base_agent import BaseAgent
from src.envs import Connect4Env, RewardConfig
from src.envs.base import TurnBasedEnv, StepResult
from src.envs.connect4 import CONNECT4_COLS, CONNECT4_ROWS
from src.envs.othello import OthelloEnv
from src.training.random_opening import RandomOpeningConfig, maybe_apply_random_opening
from src.training.trainer import StartPolicy


def agent_env_step(
    env: TurnBasedEnv,
    agent: BaseAgent,
    obs: np.ndarray,
    *,
    deterministic: bool,
) -> StepResult:
    """Perform one ``env`` transition using ``agent``. MiniBG structured PPO uses ``step_structured``."""
    if callable(getattr(agent, "act_structured", None)) and callable(
        getattr(env, "step_structured", None)
    ):
        if hasattr(agent, "set_env"):
            agent.set_env(env)
        legal_list = env.legal_structured_actions()
        struct_act, board_perm, _ = agent.act_structured(
            obs, legal_list, env, deterministic=deterministic
        )
        return env.step_structured(struct_act, board_perm=board_perm)
    if hasattr(agent, "set_env"):
        agent.set_env(env)
    action = agent.act(obs, legal_mask=env.legal_actions_mask, deterministic=deterministic)
    return env.step(int(action))


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
    Play a single game between ``agent`` and ``opponent`` on ``env``.

    Returns a dict with keys ``winner`` (1/-1/0), ``steps``, ``agent_token``,
    ``reward`` (+1/-1/0 from agent's perspective) and raw ``info`` from env.
    """
    rng = random.Random(seed) if seed is not None else None
    obs = env.reset()

    if random_opening_config is not None:
        opener_rng = rng if rng is not None else random
        obs, _, prologue_done = maybe_apply_random_opening(
            env=env, initial_obs=obs, config=random_opening_config, rng=opener_rng,
        )
        if prologue_done:
            obs = env.reset()

    agent_token = _resolve_agent_token(start_policy, rng)
    steps = 0
    done = False

    for participant in (agent, opponent):
        if hasattr(participant, "set_env"):
            participant.set_env(env)

    while not done:
        current_token = _current_player_token(env)
        is_agent_turn = current_token == agent_token
        actor = agent if is_agent_turn else opponent
        deterministic = deterministic_agent if is_agent_turn else deterministic_opponent

        step = agent_env_step(env, actor, obs, deterministic=deterministic)
        steps += 1

        if step.done:
            winner = step.info.get("winner")
            return {
                "winner": winner,
                "steps": steps,
                "agent_token": agent_token,
                "reward": _result_to_reward(winner, agent_token),
                "info": step.info,
            }

        obs = step.obs

    raise RuntimeError("Game loop exited without reaching a terminal state.")


def _resolve_agent_token(start_policy: StartPolicy, rng: Optional[random.Random]) -> int:
    if start_policy == StartPolicy.AGENT_FIRST:
        return 1
    if start_policy == StartPolicy.OPPONENT_FIRST:
        return -1
    coin = rng.random() if rng is not None else random.random()
    return 1 if coin < 0.5 else -1


def resolve_opening_agent_token(
    start_policy: StartPolicy,
    *,
    seed: Optional[int] = None,
) -> int:
    """Same opening-seat rule as :func:`play_single_game` when ``random_opening_config`` is ``None``.

    The returned token ``±1`` matches ``MiniBGEnv.current_player_token`` indexing (P0 → ``+1``).
    """
    rng = random.Random(seed) if seed is not None else None
    return _resolve_agent_token(start_policy, rng)


def _current_player_token(env: TurnBasedEnv) -> int:
    """Return +1 for the first player, -1 for the second.

    Falls back to ``current_player()`` (0/1 index) for envs that don't expose
    ``current_player_token`` (e.g. toy environments).
    """
    if hasattr(env, "current_player_token"):
        return int(getattr(env, "current_player_token"))
    if hasattr(env, "current_player"):
        return 1 if env.current_player() == 0 else -1
    raise AttributeError("Environment must expose current_player_token or current_player().")


def _result_to_reward(winner: Optional[int], agent_token: int) -> int:
    if winner is None or winner == 0:
        return 0
    return 1 if winner == agent_token else -1


def _default_env_factory(game_id: str, reward_config: RewardConfig) -> TurnBasedEnv:
    """Create environment by game_id."""
    gid = (game_id or "connect4").strip().lower()
    if gid == "othello":
        return OthelloEnv(size=8, reward_config=reward_config)
    if gid == "minibg":
        import src.envs  # noqa: F401 — register games
        from src.registry import make_game

        return make_game("minibg", reward_config=reward_config)
    return Connect4Env(rows=CONNECT4_ROWS, cols=CONNECT4_COLS, reward_config=reward_config)


def _apply_turn_batch(
    envs: List[TurnBasedEnv],
    turn_mask: List[bool],
    obs_list: List[Optional[np.ndarray]],
    agent,
    *,
    deterministic: bool = True,
) -> List[Optional[StepResult]]:
    results: List[Optional[StepResult]] = [None] * len(envs)
    indices = [i for i in range(len(envs)) if turn_mask[i]]
    if not indices:
        return results
    if hasattr(agent, "act_batch"):
        obs_batch = np.stack([obs_list[i] for i in indices])
        legal_batch = np.stack([envs[i].legal_actions_mask for i in indices])
        out = agent.act_batch(obs_batch, legal_batch, deterministic=deterministic)
        for k, i in enumerate(indices):
            results[i] = envs[i].step(int(out[k]))
    else:
        for i in indices:
            ob = obs_list[i]
            if ob is None:
                continue
            results[i] = agent_env_step(
                envs[i],
                agent,
                ob,
                deterministic=deterministic,
            )
    return results


def play_match_batched(
    agent1,
    agent2,
    num_games: int,
    batch_size: int = 32,
    seed: Optional[int] = None,
    randomize_first_player: bool = False,
    reward_config: Optional[RewardConfig] = None,
    random_opening_config: Optional[RandomOpeningConfig] = None,
    collect_episode_lengths: bool = False,
    game_id: str = "connect4",
    env_factory: Optional[Callable[[RewardConfig], TurnBasedEnv]] = None,
    deterministic: bool = True,
) -> Union[Tuple[int, int, int], Tuple[int, int, int, List[int]]]:
    """
    Play num_games between agent1 and agent2 using batch_size parallel envs.
    Batches forward passes when an agent implements act_batch (e.g. DQN).
    Same return convention as play_match.
    """
    if num_games < 1:
        raise ValueError("num_games must be >= 1")
    if batch_size < 1:
        raise ValueError(
            "play_match_batched requires batch_size >= 1 (creates that many parallel envs). "
            "For sequential play use play_match() instead."
        )
    if reward_config is None:
        reward_config = RewardConfig()
    if env_factory is not None:
        envs = [env_factory(reward_config) for _ in range(batch_size)]
    else:
        envs = [
            _default_env_factory(str(game_id), reward_config) for _ in range(batch_size)
        ]
    n = len(envs)

    obs: List[Optional[np.ndarray]] = [None] * n
    done = [True] * n
    agent1_is_player_1: List[bool] = [True] * n
    moves_count: List[int] = [0] * n
    active_game_ids: List[Optional[int]] = [None] * n
    next_game_id = 0
    agent1_wins = draws = agent2_wins = 0
    episode_lengths: List[int] = [] if collect_episode_lengths else []

    def start_game(slot: int, gid: int) -> None:
        nonlocal next_game_id
        env = envs[slot]
        game_rng = random.Random(seed + gid) if seed is not None else random
        ob = env.reset(seed=seed + gid if seed is not None else None)
        if random_opening_config is not None:
            ob, _, prologue_done = maybe_apply_random_opening(
                env=env, initial_obs=ob, config=random_opening_config, rng=game_rng
            )
            if prologue_done:
                ob = env.reset(seed=seed + gid if seed is not None else None)
        obs[slot] = ob
        done[slot] = False
        agent1_is_player_1[slot] = (game_rng.random() < 0.5) if randomize_first_player else True
        moves_count[slot] = 0
        active_game_ids[slot] = gid
        next_game_id += 1

    for i in range(min(batch_size, num_games)):
        start_game(i, i)

    while agent1_wins + draws + agent2_wins < num_games:
        agent1_turn = [
            i for i in range(n)
            if not done[i] and (_current_player_token(envs[i]) == 1) == agent1_is_player_1[i]
        ]
        agent2_turn = [
            i for i in range(n)
            if not done[i] and (_current_player_token(envs[i]) == 1) != agent1_is_player_1[i]
        ]
        turn_mask1 = [i in agent1_turn for i in range(n)]
        turn_mask2 = [i in agent2_turn for i in range(n)]
        res1 = _apply_turn_batch(envs, turn_mask1, obs, agent1, deterministic=deterministic)
        res2 = _apply_turn_batch(envs, turn_mask2, obs, agent2, deterministic=deterministic)

        for i in range(n):
            res = res1[i] if res1[i] is not None else res2[i]
            if res is None:
                continue
            obs[i] = res.obs
            moves_count[i] += 1
            if res.done:
                done[i] = True
                w = res.info.get("winner")
                a1_first = agent1_is_player_1[i]
                if w == 0:
                    draws += 1
                elif (w == 1 and a1_first) or (w == -1 and not a1_first):
                    agent1_wins += 1
                else:
                    agent2_wins += 1
                if collect_episode_lengths:
                    episode_lengths.append(moves_count[i])
                if next_game_id < num_games:
                    start_game(i, next_game_id)
                else:
                    active_game_ids[i] = None

    if collect_episode_lengths:
        return agent1_wins, draws, agent2_wins, episode_lengths
    return agent1_wins, draws, agent2_wins


def play_match(
    agent1,
    agent2,
    num_games: int = 100,
    seed: Optional[int] = None,
    randomize_first_player: bool = False,
    reward_config: Optional[RewardConfig] = None,
    random_opening_config: Optional[RandomOpeningConfig] = None,
    collect_episode_lengths: bool = False,
    env: Optional[TurnBasedEnv] = None,
    *,
    game_id: str = "connect4",
) -> Union[Tuple[int, int, int], Tuple[int, int, int, List[int]]]:
    """
    Play a match between two agents.
    
    Args:
        agent1: First agent
        agent2: Second agent
        num_games: Number of games to play
        seed: Random seed (used for reproducibility)
        randomize_first_player: If True, randomly choose who goes first each game.
                               If False, agent1 always goes first (as player 1).
        reward_config: RewardConfig object (default: RewardConfig with default values)
        random_opening_config: Optional configuration for randomized opening prologues.
        collect_episode_lengths: If True, also return episode lengths.
        env: Optional environment to use. If None, creates env from ``game_id``.
        game_id: When ``env`` is None: ``connect4``, ``othello``, or ``minibg``.
        
    Returns:
        Tuple of (agent1_wins, draws, agent2_wins). If ``collect_episode_lengths`` is True,
        also returns a list with the number of half-moves (ply) for every game played.
    """
    # Create default env if not provided
    if env is None:
        if reward_config is None:
            reward_config = RewardConfig()
        env = _default_env_factory(game_id, reward_config)

    for agent in (agent1, agent2):
        if hasattr(agent, "set_env"):
            agent.set_env(env)

    start_policy = StartPolicy.RANDOM if randomize_first_player else StartPolicy.AGENT_FIRST
    agent1_wins = draws = agent2_wins = 0
    episode_lengths: List[int] | None = [] if collect_episode_lengths else None

    for game_idx in range(num_games):
        game_seed = (seed + game_idx) if seed is not None else None
        result = play_single_game(
            env, agent1, agent2,
            start_policy=start_policy,
            random_opening_config=random_opening_config,
            deterministic_agent=False,
            deterministic_opponent=False,
            seed=game_seed,
        )
        winner = result["winner"]
        if winner == 0:
            draws += 1
        elif winner == result["agent_token"]:
            agent1_wins += 1
        else:
            agent2_wins += 1
        if episode_lengths is not None:
            episode_lengths.append(result["steps"])

    if episode_lengths is not None:
        return agent1_wins, draws, agent2_wins, episode_lengths
    return agent1_wins, draws, agent2_wins

