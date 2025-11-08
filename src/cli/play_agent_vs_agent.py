"""CLI for playing agent vs agent."""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.envs import Connect4Env
from src.agents import RandomAgent, HeuristicAgent, SmartHeuristicAgent, QLearningAgent, DQNAgent


def play_agent_vs_agent(
    agent1_path: str,
    agent1_type: str,
    agent2_path: str,
    agent2_type: str,
    num_games: int = 1,
    render: bool = True,
    seed: int = 42,
):
    """
    Play agent vs agent games.
    
    Args:
        agent1_path: Path to agent1 checkpoint
        agent1_type: Type of agent1 ('random', 'heuristic', 'qlearning', or 'dqn')
        agent2_path: Path to agent2 checkpoint
        agent2_type: Type of agent2 ('random', 'heuristic', 'qlearning', or 'dqn')
        num_games: Number of games to play
        render: Whether to render games
        seed: Random seed
    """
    env = Connect4Env(rows=6, cols=7)
    
    # Load agent1
    if agent1_type == "random":
        agent1 = RandomAgent(seed=seed)
    elif agent1_type == "heuristic":
        agent1 = HeuristicAgent(seed=seed)
    elif agent1_type == "smart_heuristic":
        agent1 = SmartHeuristicAgent(seed=seed)
    elif agent1_type == "qlearning":
        agent1 = QLearningAgent(seed=seed)
        agent1.load(agent1_path)
        agent1.eval()
    elif agent1_type == "dqn":
        agent1 = DQNAgent(rows=6, cols=7, seed=seed)
        agent1.load(agent1_path)
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
        agent2 = QLearningAgent(seed=seed + 1)
        agent2.load(agent2_path)
        agent2.eval()
    elif agent2_type == "dqn":
        agent2 = DQNAgent(rows=6, cols=7, seed=seed + 1)
        agent2.load(agent2_path)
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


def main():
    parser = argparse.ArgumentParser(description="Play agent vs agent")
    parser.add_argument("--agent1-path", type=str, default=None, help="Path to agent1 checkpoint")
    parser.add_argument("--agent1-type", type=str, choices=["random", "heuristic", "smart_heuristic", "qlearning", "dqn"], required=True, help="Agent1 type")
    parser.add_argument("--agent2-path", type=str, default=None, help="Path to agent2 checkpoint")
    parser.add_argument("--agent2-type", type=str, choices=["random", "heuristic", "smart_heuristic", "qlearning", "dqn"], required=True, help="Agent2 type")
    parser.add_argument("--num-games", type=int, default=1, help="Number of games to play")
    parser.add_argument("--no-render", action="store_true", help="Don't render games")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    if args.agent1_type in ["qlearning", "dqn"] and args.agent1_path is None:
        print("Error: --agent1-path is required for qlearning and dqn agents")
        sys.exit(1)
    
    if args.agent2_type in ["qlearning", "dqn"] and args.agent2_path is None:
        print("Error: --agent2-path is required for qlearning and dqn agents")
        sys.exit(1)
    
    play_agent_vs_agent(
        agent1_path=args.agent1_path,
        agent1_type=args.agent1_type,
        agent2_path=args.agent2_path,
        agent2_type=args.agent2_type,
        num_games=args.num_games,
        render=not args.no_render,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

