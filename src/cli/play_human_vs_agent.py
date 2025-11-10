"""CLI for playing against agent."""

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


def play_human_vs_agent(
    agent_type: Literal["random", "heuristic", "smart_heuristic", "qlearning", "dqn"],
    agent_path: Optional[str] = None,
    human_first: bool = True,
    seed: int = 42,
):
    """
    Play a game against an agent.
    
    Args:
        agent_type: Type of agent ('random', 'heuristic', 'qlearning', or 'dqn')
        agent_path: Path to agent checkpoint (required for qlearning and dqn)
        human_first: Whether human plays first
        seed: Random seed
    """
    if agent_type in ["qlearning", "dqn"] and agent_path is None:
        print("Error: agent_path is required for qlearning and dqn agents")
        sys.exit(1)
    env = Connect4Env(rows=6, cols=7)
    
    # Load agent
    if agent_type == "random":
        agent = RandomAgent(seed=seed)
    elif agent_type == "heuristic":
        agent = HeuristicAgent(seed=seed)
    elif agent_type == "smart_heuristic":
        agent = SmartHeuristicAgent(seed=seed)
    elif agent_type == "qlearning":
        agent = QLearningAgent.load(agent_path, seed=seed)
        agent.epsilon = 0.0  # Жадная политика для игры с человеком
        agent.eval()
    elif agent_type == "dqn":
        observation_builder = BoardChannels(board_shape=(6, 7))
        action_space = DiscreteActionSpace(n=7)
        agent = DQNAgent.load(
            agent_path,
            seed=seed,
            observation_shape=observation_builder.observation_shape,
            observation_type=observation_builder.observation_type,
            num_actions=action_space.size,
            action_space=action_space,
        )
        agent.epsilon = 0.0  # Жадная политика для игры с человеком
        agent.eval()
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    print("=" * 50)
    print("Connect Four - Human vs Agent")
    print("=" * 50)
    print(f"Agent type: {agent_type}")
    print(f"Human plays: {'first (X)' if human_first else 'second (O)'}")
    print("=" * 50)
    print()
    
    obs = env.reset()
    done = False
    current_player = 0 if human_first else 1  # 0: human, 1: agent
    
    while not done:
        env.render()
        
        legal_actions = env.get_legal_actions()
        
        if current_player == 0:
            # Human's turn
            print(f"Your turn! Legal columns: {legal_actions}")
            while True:
                try:
                    action = int(input("Enter column (0-6): "))
                    if action in legal_actions:
                        break
                    else:
                        print(f"Invalid action! Legal columns: {legal_actions}")
                except ValueError:
                    print("Please enter a valid number!")
        else:
            # Agent's turn
            print("Agent's turn...")
            action = agent.select_action(obs, legal_actions)
            print(f"Agent chose column: {action}")
        
        next_obs, reward, done, info = env.step(action)
        
        if done:
            env.render()
            winner = info.get("winner")
            if winner == 1:
                if human_first:
                    print("You win! 🎉")
                else:
                    print("Agent wins! 😢")
            elif winner == -1:
                if human_first:
                    print("Agent wins! 😢")
                else:
                    print("You win! 🎉")
            else:
                print("It's a draw! 🤝")
            break
        
        obs = next_obs
        current_player = 1 - current_player
        print()


if __name__ == "__main__":
    tyro.cli(play_human_vs_agent)

