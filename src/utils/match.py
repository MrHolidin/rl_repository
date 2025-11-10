"""Utilities for playing matches between agents."""

import random
import numpy as np
from typing import Tuple, Optional

from src.envs import Connect4Env, RewardConfig
from src.training.random_opening import RandomOpeningConfig, maybe_apply_random_opening


def play_match(
    agent1,
    agent2,
    num_games: int = 100,
    seed: Optional[int] = None,
    randomize_first_player: bool = False,
    reward_config: Optional[RewardConfig] = None,
    random_opening_config: Optional[RandomOpeningConfig] = None,
) -> Tuple[int, int, int]:
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
        
    Returns:
        Tuple of (agent1_wins, draws, agent2_wins)
    """
    # Initialize reward config
    if reward_config is None:
        reward_config = RewardConfig()
    
    env = Connect4Env(
        rows=6,
        cols=7,
        reward_config=reward_config,
    )
    
    agent1_wins = 0
    draws = 0
    agent2_wins = 0
    
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
        
        while not done:
            legal_actions = env.get_legal_actions()
            
            # Determine which agent should move based on env.current_player
            # env.current_player == 1 means player 1
            # env.current_player == -1 means player -1
            if env.current_player == 1:
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
                break
            
            obs = next_obs
    
    return agent1_wins, draws, agent2_wins

