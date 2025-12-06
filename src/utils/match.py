"""Utilities for playing matches between agents."""

import random
import numpy as np
from typing import List, Optional, Tuple, Union

from src.envs import Connect4Env, RewardConfig
from src.envs.base import TurnBasedEnv
from src.games.connect4 import CONNECT4_COLS, CONNECT4_ROWS
from src.training.random_opening import RandomOpeningConfig, maybe_apply_random_opening


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

