"""CLI for playing agent vs agent."""

import os
import sys
from pathlib import Path
from typing import Literal, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import tyro

from src.envs import Connect4Env
from src.agents import RandomAgent, HeuristicAgent, SmartHeuristicAgent, QLearningAgent, DQNAgent
from src.features.action_space import DiscreteActionSpace
from src.features.observation_builder import BoardChannels


def play_agent_vs_agent(
    agent1_type: Literal["random", "heuristic", "smart_heuristic", "qlearning", "dqn"],
    agent2_type: Literal["random", "heuristic", "smart_heuristic", "qlearning", "dqn"],
    agent1_path: Optional[str] = None,
    agent2_path: Optional[str] = None,
    num_games: int = 1,
    render: bool = True,
    seed: int = 42,
):
    """
    Play agent vs agent games.
    
    Args:
        agent1_type: Type of agent1 ('random', 'heuristic', 'qlearning', or 'dqn')
        agent2_type: Type of agent2 ('random', 'heuristic', 'qlearning', or 'dqn')
        agent1_path: Path to agent1 checkpoint (required for qlearning and dqn)
        agent2_path: Path to agent2 checkpoint (required for qlearning and dqn)
        num_games: Number of games to play
        render: Whether to render games
        seed: Random seed
    """
    if agent1_type in ["qlearning", "dqn"] and agent1_path is None:
        print("Error: agent1_path is required for qlearning and dqn agents")
        sys.exit(1)
    
    if agent2_type in ["qlearning", "dqn"] and agent2_path is None:
        print("Error: agent2_path is required for qlearning and dqn agents")
        sys.exit(1)
    env = Connect4Env(rows=6, cols=7)
    
    # Load agent1
    if agent1_type == "random":
        agent1 = RandomAgent(seed=seed)
    elif agent1_type == "heuristic":
        agent1 = HeuristicAgent(seed=seed)
    elif agent1_type == "smart_heuristic":
        agent1 = SmartHeuristicAgent(seed=seed)
    elif agent1_type == "qlearning":
        agent1 = QLearningAgent.load(agent1_path, seed=seed)
        agent1.eval()
    elif agent1_type == "dqn":
        observation_builder = BoardChannels(board_shape=(6, 7))
        action_space = DiscreteActionSpace(n=7)
        agent1 = DQNAgent.load(
            agent1_path,
            seed=seed,
            observation_shape=observation_builder.observation_shape,
            observation_type=observation_builder.observation_type,
            num_actions=action_space.size,
            action_space=action_space,
        )
        agent1.eval()
    else:
        raise ValueError(f"Unknown agent type: {agent1_type}")
    
    # Load agent2
    if agent2_type == "random":
        agent2 = RandomAgent(seed=seed + 1)
    elif agent2_type == "heuristic":
        agent2 = HeuristicAgent(seed=seed + 1)
    elif agent2_type == "smart_heuristic":
        agent2 = SmartHeuristicAgent(seed=seed + 1)
    elif agent2_type == "qlearning":
        agent2 = QLearningAgent.load(agent2_path, seed=seed + 1)
        agent2.eval()
    elif agent2_type == "dqn":
        observation_builder = BoardChannels(board_shape=(6, 7))
        action_space = DiscreteActionSpace(n=7)
        agent2 = DQNAgent.load(
            agent2_path,
            seed=seed + 1,
            observation_shape=observation_builder.observation_shape,
            observation_type=observation_builder.observation_type,
            num_actions=action_space.size,
            action_space=action_space,
        )
        agent2.eval()
    else:
        raise ValueError(f"Unknown agent type: {agent2_type}")
    
    print("=" * 50)
    print("Connect Four - Agent vs Agent")
    print("=" * 50)
    print(f"Agent 1: {agent1_type}")
    print(f"Agent 2: {agent2_type}")
    print(f"Games: {num_games}")
    print("=" * 50)
    print()
    
    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    
    for game in range(num_games):
        obs = env.reset()
        done = False
        current_player = 0  # 0: agent1, 1: agent2
        
        if render:
            print(f"\nGame {game + 1}/{num_games}")
            print("-" * 50)
        
        while not done:
            legal_actions = env.get_legal_actions()
            
            if current_player == 0:
                action = agent1.select_action(obs, legal_actions)
            else:
                action = agent2.select_action(obs, legal_actions)
            
            next_obs, reward, done, info = env.step(action)
            
            if render:
                env.render()
            
            if done:
                winner = info.get("winner")
                if winner == 1:  # Agent1 is player 1
                    agent1_wins += 1
                    if render:
                        print("Agent 1 wins!")
                elif winner == -1:  # Agent2 is player -1
                    agent2_wins += 1
                    if render:
                        print("Agent 2 wins!")
                else:
                    draws += 1
                    if render:
                        print("Draw!")
                break
            
            obs = next_obs
            current_player = 1 - current_player
        
        if render:
            print()
    
    # Print summary
    print("=" * 50)
    print("Results Summary")
    print("=" * 50)
    print(f"Agent 1 wins: {agent1_wins} ({agent1_wins/num_games*100:.1f}%)")
    print(f"Agent 2 wins: {agent2_wins} ({agent2_wins/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")
    print("=" * 50)


if __name__ == "__main__":
    tyro.cli(play_agent_vs_agent)

