"""Evaluation script for agents."""

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.envs import Connect4Env
from src.agents import RandomAgent, HeuristicAgent, SmartHeuristicAgent, QLearningAgent, DQNAgent
import matplotlib.pyplot as plt
import numpy as np


def evaluate_agent_vs_opponent(
    agent_path: str,
    agent_type: str,
    opponent_type: str = "random",
    num_episodes: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Evaluate agent against opponent.
    
    Args:
        agent_path: Path to agent checkpoint
        agent_type: Type of agent ('qlearning' or 'dqn')
        opponent_type: Type of opponent ('random' or 'heuristic')
        num_episodes: Number of evaluation episodes
        seed: Random seed
        
    Returns:
        Tuple of (win_rate, draw_rate, loss_rate)
    """
    env = Connect4Env(rows=6, cols=7)
    
    # Load agent
    if agent_type == "qlearning":
        agent = QLearningAgent(seed=seed)
        agent.load(agent_path)
    elif agent_type == "dqn":
        agent = DQNAgent(rows=6, cols=7, seed=seed)
        agent.load(agent_path)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    agent.eval()
    
    # Create opponent
    if opponent_type == "random":
        opponent = RandomAgent(seed=seed + 1)
    elif opponent_type == "heuristic":
        opponent = HeuristicAgent(seed=seed + 1)
    elif opponent_type == "smart_heuristic":
        opponent = SmartHeuristicAgent(seed=seed + 1)
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")
    
    wins = 0
    draws = 0
    losses = 0
    episode_lengths = []
    
    print(f"Evaluating {agent_type} agent against {opponent_type} opponent...")
    print(f"Episodes: {num_episodes}")
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        current_player = 0  # 0: agent, 1: opponent
        episode_length = 0
        
        while not done:
            legal_actions = env.get_legal_actions()
            
            if current_player == 0:
                action = agent.select_action(obs, legal_actions)
            else:
                action = opponent.select_action(obs, legal_actions)
            
            next_obs, reward, done, info = env.step(action)
            
            if done:
                winner = info.get("winner")
                if winner == 1:  # Agent is player 1
                    wins += 1
                elif winner == -1:  # Opponent is player -1
                    losses += 1
                else:
                    draws += 1
                break
            
            obs = next_obs
            current_player = 1 - current_player
            episode_length += 1
        
        episode_lengths.append(episode_length)
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}")
    
    win_rate = wins / num_episodes
    draw_rate = draws / num_episodes
    loss_rate = losses / num_episodes
    
    print(f"\nResults:")
    print(f"Win rate: {win_rate:.2%} ({wins}/{num_episodes})")
    print(f"Draw rate: {draw_rate:.2%} ({draws}/{num_episodes})")
    print(f"Loss rate: {loss_rate:.2%} ({losses}/{num_episodes})")
    print(f"Average episode length: {np.mean(episode_lengths):.2f}")
    
    return win_rate, draw_rate, loss_rate


def plot_results(results: dict, save_path: str = None):
    """
    Plot evaluation results.
    
    Args:
        results: Dictionary of results
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Win/Draw/Loss rates
    ax1 = axes[0]
    labels = list(results.keys())
    win_rates = [results[label]["win_rate"] for label in labels]
    draw_rates = [results[label]["draw_rate"] for label in labels]
    loss_rates = [results[label]["loss_rate"] for label in labels]
    
    x = np.arange(len(labels))
    width = 0.25
    
    ax1.bar(x - width, win_rates, width, label="Win", color="green")
    ax1.bar(x, draw_rates, width, label="Draw", color="gray")
    ax1.bar(x + width, loss_rates, width, label="Loss", color="red")
    
    ax1.set_xlabel("Opponent")
    ax1.set_ylabel("Rate")
    ax1.set_title("Win/Draw/Loss Rates")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.set_ylim([0, 1])
    
    # Bar plot
    ax2 = axes[1]
    categories = ["Win", "Draw", "Loss"]
    values = [
        sum(results[label]["win_rate"] for label in labels) / len(labels),
        sum(results[label]["draw_rate"] for label in labels) / len(labels),
        sum(results[label]["loss_rate"] for label in labels) / len(labels),
    ]
    
    ax2.bar(categories, values, color=["green", "gray", "red"])
    ax2.set_ylabel("Average Rate")
    ax2.set_title("Overall Results")
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate agent")
    parser.add_argument("--agent-path", type=str, required=True, help="Path to agent checkpoint")
    parser.add_argument("--agent-type", type=str, choices=["qlearning", "dqn"], required=True, help="Agent type")
    parser.add_argument("--opponent-type", type=str, choices=["random", "heuristic", "smart_heuristic"], default="random", help="Opponent type")
    parser.add_argument("--num-episodes", type=int, default=1000, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--plot", action="store_true", help="Plot results")
    parser.add_argument("--plot-path", type=str, default=None, help="Path to save plot")
    
    args = parser.parse_args()
    
    win_rate, draw_rate, loss_rate = evaluate_agent_vs_opponent(
        agent_path=args.agent_path,
        agent_type=args.agent_type,
        opponent_type=args.opponent_type,
        num_episodes=args.num_episodes,
        seed=args.seed,
    )
    
    if args.plot:
        results = {
            args.opponent_type: {
                "win_rate": win_rate,
                "draw_rate": draw_rate,
                "loss_rate": loss_rate,
            }
        }
        plot_results(results, save_path=args.plot_path)

