"""Visualize a game between a trained model and a bot in the console."""

import sys
import time
from pathlib import Path
from typing import Literal, Optional
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import tyro

from src.envs import Connect4Env
from src.agents import RandomAgent, HeuristicAgent, SmartHeuristicAgent, QLearningAgent, DQNAgent


def find_latest_checkpoint(
    checkpoint_dir: str = "data/checkpoints",
    model_type: Literal["dqn", "qlearning"] = "dqn",
) -> Optional[str]:
    """
    Find the latest checkpoint from all training runs.
    
    Args:
        checkpoint_dir: Directory to search for checkpoints
        model_type: Type of model ('dqn' or 'qlearning')
        
    Returns:
        Path to latest checkpoint, or None if not found
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None
    
    # Determine pattern based on model type
    if model_type == "dqn":
        pattern = "*.pt"
    elif model_type == "qlearning":
        pattern = "*.pkl"
    else:
        return None
    
    # Find all matching files recursively
    checkpoints = list(checkpoint_path.rglob(pattern))
    
    if not checkpoints:
        return None
    
    # Filter out intermediate files (like "stopped" or "final")
    # Keep only numbered checkpoints (e.g., "dqn_episode_1000.pt")
    filtered = []
    for cp in checkpoints:
        name = cp.stem
        # Skip final and stopped checkpoints, keep only numbered ones
        if "final" not in name.lower() and "stopped" not in name.lower():
            # Extract episode number from filename
            # Format: "dqn_episode_1000" or "qlearning_episode_1000"
            try:
                if "_episode_" in name:
                    episode_str = name.split("_episode_")[-1]
                    episode_num = int(episode_str)
                    filtered.append((episode_num, str(cp)))
            except (ValueError, IndexError):
                # Skip files that don't match the pattern
                continue
    
    if not filtered:
        return None
    
    # Sort by episode number and take the latest one
    filtered.sort(key=lambda x: x[0])
    return filtered[-1][1]  # Return path of latest checkpoint


def visualize_game(
    model_path: Optional[str] = None,
    model_type: Literal["qlearning", "dqn"] = "dqn",
    opponent_type: Literal["random", "heuristic", "smart_heuristic"] = "random",
    opponent_first: bool = False,
    use_epsilon: bool = False,
    delay: float = 1.0,
    seed: int = 42,
    checkpoint_dir: str = "data/checkpoints",
    use_latest: bool = False,
):
    """
    Visualize a game between a trained model and a bot.
    
    Args:
        model_path: Path to model checkpoint (if None and use_latest=True, finds latest automatically)
        model_type: Type of model ('qlearning' or 'dqn')
        opponent_type: Type of opponent ('random', 'heuristic', or 'smart_heuristic')
        opponent_first: Whether opponent plays first (model plays second)
        use_epsilon: Whether to use epsilon-greedy (False = pure exploitation)
        delay: Delay between moves in seconds
        seed: Random seed
        checkpoint_dir: Directory to search for checkpoints (if use_latest=True)
        use_latest: If True, automatically find and use latest checkpoint
    """
    # Auto-find latest checkpoint if requested
    if use_latest or model_path is None:
        print(f"Searching for latest {model_type} checkpoint in {checkpoint_dir}...")
        model_path = find_latest_checkpoint(checkpoint_dir, model_type)
        if model_path is None:
            raise ValueError(f"No {model_type} checkpoints found in {checkpoint_dir}")
        print(f"✓ Found latest checkpoint: {model_path}")
        print()
    model_first = not opponent_first
    env = Connect4Env(rows=6, cols=7, reward_config=None)  # Use default RewardConfig
    
    # Load model
    if model_type == "qlearning":
        model = QLearningAgent(seed=seed)
        model.load(model_path)
    elif model_type == "dqn":
        # Auto-detect network_type from checkpoint
        network_type = DQNAgent.get_network_type_from_checkpoint(model_path)
        model = DQNAgent(rows=6, cols=7, seed=seed, network_type=network_type)
        model.load(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Set epsilon
    if not use_epsilon:
        if hasattr(model, 'epsilon'):
            model.epsilon = 0.0
        model.eval()
    else:
        model.train()
    
    # Create opponent
    if opponent_type == "random":
        opponent = RandomAgent(seed=seed + 1)
    elif opponent_type == "heuristic":
        opponent = HeuristicAgent(seed=seed + 1)
    elif opponent_type == "smart_heuristic":
        opponent = SmartHeuristicAgent(seed=seed + 1)
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")
    
    # Get epsilon for display
    epsilon = getattr(model, 'epsilon', None)
    
    print("=" * 60)
    print("Connect Four - Model vs Bot Visualization")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Model type: {model_type}")
    print(f"Opponent: {opponent_type}")
    print(f"Model plays: {'first (X)' if model_first else 'second (O)'}")
    if epsilon is not None:
        print(f"Model epsilon: {epsilon:.6f} {'(pure exploitation)' if epsilon == 0.0 else ''}")
    print(f"Move delay: {delay:.1f}s")
    print("=" * 60)
    print()
    
    obs = env.reset()
    done = False
    current_player = 0 if model_first else 1  # 0: model, 1: opponent
    move_count = 0
    
    while not done:
        # Clear screen for better visualization (optional)
        # print("\033[2J\033[H")  # Uncomment for screen clearing
        
        env.render()
        
        legal_actions = env.get_legal_actions()
        move_count += 1
        
        if current_player == 0:
            # Model's turn
            player_name = "Model"
            player_symbol = "X" if model_first else "O"
            print(f"\nMove {move_count}: {player_name} ({player_symbol})")
            print(f"Legal columns: {legal_actions}")
            
            action = model.select_action(obs, legal_actions)
            print(f"→ {player_name} chose column: {action}")
        else:
            # Opponent's turn
            player_name = opponent_type.capitalize()
            player_symbol = "O" if model_first else "X"
            print(f"\nMove {move_count}: {player_name} ({player_symbol})")
            print(f"Legal columns: {legal_actions}")
            
            action = opponent.select_action(obs, legal_actions)
            print(f"→ {player_name} chose column: {action}")
        
        next_obs, reward, done, info = env.step(action)
        
        if done:
            time.sleep(delay)
            # Clear and show final board
            # print("\033[2J\033[H")  # Uncomment for screen clearing
            env.render()
            
            winner = info.get("winner")
            print("\n" + "=" * 60)
            print("GAME OVER")
            print("=" * 60)
            
            if winner == 1:
                if model_first:
                    print("🏆 Model (X) wins!")
                else:
                    print(f"🏆 {opponent_type.capitalize()} (X) wins!")
            elif winner == -1:
                if model_first:
                    print(f"🏆 {opponent_type.capitalize()} (O) wins!")
                else:
                    print("🏆 Model (O) wins!")
            else:
                print("🤝 It's a draw!")
            
            print(f"Total moves: {move_count}")
            print("=" * 60)
            break
        
        obs = next_obs
        current_player = 1 - current_player
        
        # Delay between moves
        if delay > 0:
            time.sleep(delay)
        
        print()


if __name__ == "__main__":
    tyro.cli(visualize_game)

