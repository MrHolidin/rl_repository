"""CLI for playing against agent."""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.envs import Connect4Env
from src.agents import RandomAgent, HeuristicAgent, SmartHeuristicAgent, QLearningAgent, DQNAgent


def play_human_vs_agent(
    agent_path: str,
    agent_type: str,
    human_first: bool = True,
    seed: int = 42,
):
    """
    Play a game against an agent.
    
    Args:
        agent_path: Path to agent checkpoint
        agent_type: Type of agent ('random', 'heuristic', 'qlearning', or 'dqn')
        human_first: Whether human plays first
        seed: Random seed
    """
    env = Connect4Env(rows=6, cols=7)
    
    # Load agent
    if agent_type == "random":
        agent = RandomAgent(seed=seed)
    elif agent_type == "heuristic":
        agent = HeuristicAgent(seed=seed)
    elif agent_type == "smart_heuristic":
        agent = SmartHeuristicAgent(seed=seed)
    elif agent_type == "qlearning":
        agent = QLearningAgent(seed=seed)
        agent.load(agent_path)
        agent.eval()
    elif agent_type == "dqn":
        agent = DQNAgent(rows=6, cols=7, seed=seed)
        agent.load(agent_path)
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


def main():
    parser = argparse.ArgumentParser(description="Play against agent")
    parser.add_argument("--agent-path", type=str, default=None, help="Path to agent checkpoint")
    parser.add_argument("--agent-type", type=str, choices=["random", "heuristic", "smart_heuristic", "qlearning", "dqn"], required=True, help="Agent type")
    parser.add_argument("--human-first", action="store_true", help="Human plays first")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    if args.agent_type in ["qlearning", "dqn"] and args.agent_path is None:
        print("Error: --agent-path is required for qlearning and dqn agents")
        sys.exit(1)
    
    play_human_vs_agent(
        agent_path=args.agent_path,
        agent_type=args.agent_type,
        human_first=args.human_first,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

