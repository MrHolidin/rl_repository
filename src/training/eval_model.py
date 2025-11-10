"""Evaluate a trained model against multiple opponents."""

import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import tyro

from src.agents import RandomAgent, HeuristicAgent, SmartHeuristicAgent, QLearningAgent, DQNAgent
from src.features.action_space import DiscreteActionSpace
from src.features.observation_builder import BoardChannels
from src.utils.match import play_match


def evaluate_model_against_opponents(
    model_path: str,
    model_type: Literal["qlearning", "dqn"],
    opponents: Optional[List[Literal["random", "heuristic", "smart_heuristic"]]] = None,
    num_episodes: int = 1000,
    seed: int = 42,
    use_epsilon: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a model against multiple opponents.
    
    Args:
        model_path: Path to model checkpoint
        model_type: Type of model ('qlearning' or 'dqn')
        opponents: List of opponent types to test against (default: all)
        num_episodes: Number of episodes per opponent
        seed: Random seed
        use_epsilon: Whether to use epsilon-greedy (default: True)
        
    Returns:
        Dictionary with results for each opponent
    """
    if opponents is None:
        opponents = ["random", "heuristic", "smart_heuristic"]
    # Load model
    observation_builder = BoardChannels(board_shape=(6, 7))
    action_space = DiscreteActionSpace(n=7)
    if model_type == "qlearning":
        model = QLearningAgent.load(model_path, seed=seed)
    elif model_type == "dqn":
        model = DQNAgent.load(
            model_path,
            seed=seed,
            observation_shape=observation_builder.observation_shape,
            observation_type=observation_builder.observation_type,
            num_actions=action_space.size,
            action_space=action_space,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Set epsilon to 0 if use_epsilon is False (pure exploitation)
    if not use_epsilon:
        if hasattr(model, 'epsilon'):
            model.epsilon = 0.0
        # Pure exploitation: use eval mode (no epsilon, no training)
        model.eval()
    else:
        # If we want to use epsilon, we need to be in train mode
        # because select_action only applies epsilon when training=True
        model.train()
    
    # Get epsilon if available
    epsilon = getattr(model, 'epsilon', None)
    
    results = {}
    
    print("=" * 80)
    print(f"Evaluating {model_type} model: {model_path}")
    print("=" * 80)
    print(f"Episodes per opponent: {num_episodes}")
    if epsilon is not None:
        print(f"Epsilon: {epsilon:.6f} {'(pure exploitation)' if epsilon == 0.0 else ''}")
    print()
    
    for opponent_type in opponents:
        print(f"Testing against {opponent_type}...", end=" ", flush=True)
        
        # Create opponent
        if opponent_type == "random":
            opponent = RandomAgent(seed=seed + 1)
        elif opponent_type == "heuristic":
            opponent = HeuristicAgent(seed=seed + 1)
        elif opponent_type == "smart_heuristic":
            opponent = SmartHeuristicAgent(seed=seed + 1)
        else:
            raise ValueError(f"Unknown opponent type: {opponent_type}")
        
        # Use common play_match function with randomization
        # model is agent1, opponent is agent2
        # Returns (agent1_wins, draws, agent2_wins) = (model_wins, draws, opponent_wins)
        model_wins, draws, opponent_wins = play_match(
            model,
            opponent,
            num_games=num_episodes,
            seed=seed,
            randomize_first_player=True,  # Randomize who goes first each game
        )
        
        wins = model_wins
        losses = opponent_wins
        
        win_rate = wins / num_episodes
        draw_rate = draws / num_episodes
        loss_rate = losses / num_episodes
        
        results[opponent_type] = {
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": win_rate,
            "draw_rate": draw_rate,
            "loss_rate": loss_rate,
        }
        
        print(f"Done! Win: {win_rate:.1%} | Draw: {draw_rate:.1%} | Loss: {loss_rate:.1%}")
    
    model.train()
    
    return results


def print_results(results: Dict[str, Dict[str, float]], model_path: str, model=None):
    """
    Print evaluation results in a nice format.
    
    Args:
        results: Dictionary with results for each opponent
        model_path: Path to model
        model: Model object (optional, for displaying epsilon)
    """
    print()
    print("=" * 80)
    print("Results Summary")
    print("=" * 80)
    print(f"Model: {model_path}")
    if model is not None:
        epsilon = getattr(model, 'epsilon', None)
        if epsilon is not None:
            print(f"Epsilon: {epsilon:.6f}")
    print()
    
    print(f"{'Opponent':<20} {'Wins':<10} {'Draws':<10} {'Losses':<10} {'Win Rate':<15} {'Draw Rate':<15} {'Loss Rate':<15}")
    print("-" * 80)
    
    for opponent_type, stats in results.items():
        print(
            f"{opponent_type:<20} "
            f"{stats['wins']:<10} "
            f"{stats['draws']:<10} "
            f"{stats['losses']:<10} "
            f"{stats['win_rate']:>14.1%} "
            f"{stats['draw_rate']:>14.1%} "
            f"{stats['loss_rate']:>14.1%}"
        )
    
    print()
    print("=" * 80)
    print("Detailed Statistics")
    print("=" * 80)
    print()
    
    for opponent_type, stats in results.items():
        print(f"Against {opponent_type}:")
        print(f"  Wins: {stats['wins']} ({stats['win_rate']:.1%})")
        print(f"  Draws: {stats['draws']} ({stats['draw_rate']:.1%})")
        print(f"  Losses: {stats['losses']} ({stats['loss_rate']:.1%})")
        print()


def main(
    model_path: str,
    model_type: Literal["qlearning", "dqn"],
    opponents: Optional[List[Literal["random", "heuristic", "smart_heuristic"]]] = None,
    num_episodes: int = 1000,
    seed: int = 42,
    no_epsilon: bool = False,
):
    """Main entry point for evaluation."""
    # Load model to get epsilon for display
    observation_builder = BoardChannels(board_shape=(6, 7))
    action_space = DiscreteActionSpace(n=7)
    if model_type == "qlearning":
        model_for_display = QLearningAgent.load(model_path, seed=seed)
    elif model_type == "dqn":
        model_for_display = DQNAgent.load(
            model_path,
            seed=seed,
            observation_shape=observation_builder.observation_shape,
            observation_type=observation_builder.observation_type,
            num_actions=action_space.size,
            action_space=action_space,
        )
    else:
        model_for_display = None
    
    # Set epsilon to 0 if --no-epsilon flag is set
    if no_epsilon and model_for_display is not None:
        if hasattr(model_for_display, 'epsilon'):
            model_for_display.epsilon = 0.0
    
    results = evaluate_model_against_opponents(
        model_path=model_path,
        model_type=model_type,
        opponents=opponents,
        num_episodes=num_episodes,
        seed=seed,
        use_epsilon=not no_epsilon,
    )
    
    print_results(results, model_path, model=model_for_display)


if __name__ == "__main__":
    tyro.cli(main)

