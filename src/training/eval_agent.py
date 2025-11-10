"""Evaluation script for agents."""

import os
import sys
from pathlib import Path
from typing import Tuple, Literal, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import tyro

from src.agents import RandomAgent, HeuristicAgent, SmartHeuristicAgent, QLearningAgent, DQNAgent
from src.features.action_space import DiscreteActionSpace
from src.features.observation_builder import BoardChannels
from src.utils.match import play_match
import matplotlib.pyplot as plt
import numpy as np


def evaluate_agent_vs_opponent(
    agent_path: str,
    agent_type: Literal["qlearning", "dqn"],
    opponent_type: Literal["random", "heuristic", "smart_heuristic"] = "random",
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
    # Load agent
    observation_builder = BoardChannels(board_shape=(6, 7))
    action_space = DiscreteActionSpace(n=7)
    if agent_type == "qlearning":
        agent = QLearningAgent.load(agent_path, seed=seed)
    elif agent_type == "dqn":
        agent = DQNAgent.load(
            agent_path,
            seed=seed,
            observation_shape=observation_builder.observation_shape,
            observation_type=observation_builder.observation_type,
            num_actions=action_space.size,
            action_space=action_space,
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Сохраняем старое значение epsilon и устанавливаем 0 для жадной политики при оценке
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
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
    
    print(f"Evaluating {agent_type} agent against {opponent_type} opponent...")
    print(f"Episodes: {num_episodes}")
    
    # Use common play_match function with randomization
    # agent is agent1, opponent is agent2
    # Returns (agent1_wins, draws, agent2_wins) = (agent_wins, draws, opponent_wins)
    agent_wins, draws, opponent_wins = play_match(
        agent,
        opponent,
        num_games=num_episodes,
        seed=seed,
        randomize_first_player=True,  # Randomize who goes first each game
    )
    
    wins = agent_wins
    losses = opponent_wins
    
    win_rate = wins / num_episodes
    draw_rate = draws / num_episodes
    loss_rate = losses / num_episodes
    
    # Note: episode_lengths is not collected when using play_match
    # If needed, it can be added to play_match function in the future
    episode_lengths = []
    
    # Восстанавливаем старое значение epsilon
    agent.epsilon = old_epsilon
    agent.train()
    
    print(f"\nResults:")
    print(f"Win rate: {win_rate:.2%} ({wins}/{num_episodes})")
    print(f"Draw rate: {draw_rate:.2%} ({draws}/{num_episodes})")
    print(f"Loss rate: {loss_rate:.2%} ({losses}/{num_episodes})")
    if episode_lengths:
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


def main(
    agent_path: str,
    agent_type: Literal["qlearning", "dqn"],
    opponent_type: Literal["random", "heuristic", "smart_heuristic"] = "random",
    num_episodes: int = 1000,
    seed: int = 42,
    plot: bool = False,
    plot_path: Optional[str] = None,
):
    """Main entry point for evaluation."""
    win_rate, draw_rate, loss_rate = evaluate_agent_vs_opponent(
        agent_path=agent_path,
        agent_type=agent_type,
        opponent_type=opponent_type,
        num_episodes=num_episodes,
        seed=seed,
    )
    
    if plot:
        results = {
            opponent_type: {
                "win_rate": win_rate,
                "draw_rate": draw_rate,
                "loss_rate": loss_rate,
            }
        }
        plot_results(results, save_path=plot_path)


if __name__ == "__main__":
    tyro.cli(main)

