"""Utilities for playing matches between agents."""

import random
import numpy as np
from typing import Callable, List, Optional, Tuple, Union

from src.envs import Connect4Env, RewardConfig
from src.envs.base import TurnBasedEnv, StepResult
from src.envs.connect4 import CONNECT4_COLS, CONNECT4_ROWS
from src.envs.othello import OthelloEnv
from src.training.random_opening import RandomOpeningConfig, maybe_apply_random_opening


def _default_env_factory(game_id: str, reward_config: RewardConfig) -> TurnBasedEnv:
    """Create environment by game_id."""
    if game_id == "othello":
        return OthelloEnv(size=8, reward_config=reward_config)
    else:
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
            legal = envs[i].get_legal_actions()
            results[i] = envs[i].step(agent.select_action(obs_list[i], legal))
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
    if reward_config is None:
        reward_config = RewardConfig()
    if env_factory is not None:
        envs = [env_factory(reward_config) for _ in range(batch_size)]
    else:
        envs = [_default_env_factory(game_id, reward_config) for _ in range(batch_size)]
    n = len(envs)

    obs: List[Optional[np.ndarray]] = [None] * n
    done = [True] * n
    agent1_is_player_1: List[bool] = [True] * n
    moves_count: List[int] = [0] * n
    game_id: List[Optional[int]] = [None] * n
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
        game_id[slot] = gid
        next_game_id += 1

    for i in range(min(batch_size, num_games)):
        start_game(i, i)

    while agent1_wins + draws + agent2_wins < num_games:
        agent1_turn = [
            i for i in range(n)
            if not done[i] and (
                (envs[i].current_player_token == 1 and agent1_is_player_1[i])
                or (envs[i].current_player_token == -1 and not agent1_is_player_1[i])
            )
        ]
        agent2_turn = [
            i for i in range(n)
            if not done[i] and (
                (envs[i].current_player_token == 1 and not agent1_is_player_1[i])
                or (envs[i].current_player_token == -1 and agent1_is_player_1[i])
            )
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
                gid = game_id[i]
                if next_game_id < num_games:
                    start_game(i, next_game_id)
                else:
                    game_id[i] = None

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
        env: Optional environment to use. If None, creates a default Connect4Env.
        
    Returns:
        Tuple of (agent1_wins, draws, agent2_wins). If ``collect_episode_lengths`` is True,
        also returns a list with the number of half-moves (ply) for every game played.
    """
    # Create default env if not provided
    if env is None:
        if reward_config is None:
            reward_config = RewardConfig()
        env = Connect4Env(
            rows=CONNECT4_ROWS,
            cols=CONNECT4_COLS,
            reward_config=reward_config,
        )
    
    # Allow policy-based agents to access the env for state queries
    for agent in (agent1, agent2):
        if hasattr(agent, "set_env"):
            agent.set_env(env)
    
    agent1_wins = 0
    draws = 0
    agent2_wins = 0
    episode_lengths: List[int] | None = [] if collect_episode_lengths else None
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    for game_idx in range(num_games):
        # Set seed for each game for reproducibility
        if seed is not None:
            random.seed(seed + game_idx)
            np.random.seed(seed + game_idx)
        
        obs = env.reset()
        if random_opening_config is not None:
            obs, _, prologue_done = maybe_apply_random_opening(
                env=env,
                initial_obs=obs,
                config=random_opening_config,
                rng=random,
            )
            if prologue_done:
                # Rare case: random prologue finished the game. Restart without prologue.
                obs = env.reset()
        
        done = False
        
        # Determine who goes first
        if randomize_first_player:
            agent1_is_player_1 = random.random() < 0.5
        else:
            agent1_is_player_1 = True  # agent1 always goes first
        
        moves = 0
        while not done:
            legal_actions = env.get_legal_actions()
            
            # Determine which agent should move based on env.current_player_token
            # current_player_token == 1 means player 1
            # current_player_token == -1 means player -1
            current_token = env.current_player_token
            if current_token == 1:
                # Player 1's turn
                if agent1_is_player_1:
                    action = agent1.select_action(obs, legal_actions)
                else:
                    action = agent2.select_action(obs, legal_actions)
            else:
                # Player -1's turn
                if agent1_is_player_1:
                    action = agent2.select_action(obs, legal_actions)
                else:
                    action = agent1.select_action(obs, legal_actions)
            
            next_obs, reward, done, info = env.step(action)
            moves += 1
            
            if done:
                winner = info.get("winner")
                # winner == 1 means player 1 won
                # winner == -1 means player -1 won
                # winner == 0 means draw
                if winner == 0:
                    draws += 1
                elif (winner == 1 and agent1_is_player_1) or (winner == -1 and not agent1_is_player_1):
                    # Agent1 won
                    agent1_wins += 1
                else:
                    # Agent2 won
                    agent2_wins += 1
                if episode_lengths is not None:
                    episode_lengths.append(moves)
                break
            
            obs = next_obs
    
    if episode_lengths is not None:
        return agent1_wins, draws, agent2_wins, episode_lengths
    return agent1_wins, draws, agent2_wins

