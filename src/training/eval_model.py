"""Evaluate a trained model against multiple opponents."""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.envs import Connect4Env
from src.agents import RandomAgent, HeuristicAgent, SmartHeuristicAgent, QLearningAgent, DQNAgent


def evaluate_model_against_opponents(
    model_path: str,
    model_type: str,
    opponents: List[str],
    num_episodes: int = 1000,
    seed: int = 42,
    use_epsilon: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a model against multiple opponents.
    
    Args:
        model_path: Path to model checkpoint
        model_type: Type of model ('qlearning' or 'dqn')
        opponents: List of opponent types to test against
        num_episodes: Number of episodes per opponent
        seed: Random seed
        
    Returns:
        Dictionary with results for each opponent
    """
    env = Connect4Env(rows=6, cols=7)
    
    # Load model
    if model_type == "qlearning":
        model = QLearningAgent(seed=seed)
        model.load(model_path)
    elif model_type == "dqn":
        model = DQNAgent(rows=6, cols=7, seed=seed)
        model.load(model_path)
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
        
        wins = 0
        draws = 0
        losses = 0
        
        for episode in range(num_episodes):
            obs = env.reset()
            done = False
            current_player = 0  # 0: model, 1: opponent
            
            while not done:
                legal_actions = env.get_legal_actions()
                
                if current_player == 0:
                    action = model.select_action(obs, legal_actions)
                else:
                    action = opponent.select_action(obs, legal_actions)
                
                next_obs, reward, done, info = env.step(action)
                
                if done:
                    winner = info.get("winner")
                    if winner == 1:  # Model is player 1
                        wins += 1
                    elif winner == -1:  # Opponent is player -1
                        losses += 1
                    else:
                        draws += 1
                    break
                
                obs = next_obs
                current_player = 1 - current_player
        
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model against multiple opponents")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model-type", type=str, choices=["qlearning", "dqn"], required=True, help="Model type")
    parser.add_argument("--opponents", type=str, nargs="+", default=["random", "heuristic", "smart_heuristic"], 
                        choices=["random", "heuristic", "smart_heuristic"],
                        help="List of opponents to test against")
    parser.add_argument("--num-episodes", type=int, default=1000, help="Number of episodes per opponent")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-epsilon", action="store_true", help="Set epsilon to 0 (pure exploitation, no random actions)")
    
    args = parser.parse_args()
    
    # Load model to get epsilon for display
    if args.model_type == "qlearning":
        model_for_display = QLearningAgent(seed=args.seed)
        model_for_display.load(args.model_path)
    elif args.model_type == "dqn":
        model_for_display = DQNAgent(rows=6, cols=7, seed=args.seed)
        model_for_display.load(args.model_path)
    else:
        model_for_display = None
    
    # Set epsilon to 0 if --no-epsilon flag is set
    if args.no_epsilon and model_for_display is not None:
        if hasattr(model_for_display, 'epsilon'):
            model_for_display.epsilon = 0.0
    
    results = evaluate_model_against_opponents(
        model_path=args.model_path,
        model_type=args.model_type,
        opponents=args.opponents,
        num_episodes=args.num_episodes,
        seed=args.seed,
        use_epsilon=not args.no_epsilon,
    )
    
    print_results(results, args.model_path, model=model_for_display)

