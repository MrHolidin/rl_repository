"""Tournament between all trained models and heuristic agents."""

import sys
import glob
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not installed. Using matplotlib only for visualization.")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import tyro

from src.envs import Connect4Env, RewardConfig
from src.agents import DQNAgent, QLearningAgent, RandomAgent, HeuristicAgent, SmartHeuristicAgent
from src.utils.match import play_match
from src.training.random_opening import RandomOpeningConfig


def find_all_checkpoints(
    checkpoint_dir: str = "data/checkpoints",
    model_type: Optional[Literal["dqn", "qlearning"]] = None,
) -> List[str]:
    """
    Find all checkpoint files in the directory.
    From each training run (subdirectory), takes only the latest checkpoint.
    
    Args:
        checkpoint_dir: Directory to search for checkpoints
        model_type: Filter by model type ('dqn' or 'qlearning'), None for both
        
    Returns:
        List of checkpoint file paths (one per training run)
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return []
    
    # Determine patterns based on model type
    if model_type == "dqn":
        patterns = ["*.pt"]
    elif model_type == "qlearning":
        patterns = ["*.pkl"]
    else:
        # Search for both types
        patterns = ["*.pt", "*.pkl"]
    
    # Find all matching files recursively
    checkpoints = []
    for pattern in patterns:
        checkpoints.extend(checkpoint_path.rglob(pattern))
    
    # Group checkpoints by directory (each directory = one training run)
    checkpoints_by_dir = defaultdict(list)
    
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
                    # Use parent directory as key (training run)
                    parent_dir = str(cp.parent)
                    checkpoints_by_dir[parent_dir].append((episode_num, str(cp)))
            except (ValueError, IndexError):
                # Skip files that don't match the pattern
                continue
    
    # From each directory, take only the latest checkpoint (highest episode number)
    latest_checkpoints = []
    for parent_dir, cp_list in checkpoints_by_dir.items():
        if cp_list:
            # Sort by episode number and take the last one
            cp_list.sort(key=lambda x: x[0])
            latest_checkpoints.append(cp_list[-1][1])  # Take path, not episode number
    
    return sorted(latest_checkpoints)


def load_agent_from_checkpoint(
    checkpoint_path: str,
    model_type: str,
    device: Optional[str] = None,
    seed: int = 42,
    epsilon: float = 0.0,
    use_folder_name: bool = True,
) -> Tuple[DQNAgent | QLearningAgent, str]:
    """
    Load an agent from a checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_type: Type of model ('dqn' or 'qlearning')
        device: Device to use for DQN ('cuda' or 'cpu')
        seed: Random seed
        epsilon: Epsilon value (default: 0.0 for pure exploitation)
        use_folder_name: If True, use folder name as agent name; otherwise use file name
        
    Returns:
        Tuple of (agent, agent_name)
    """
    if use_folder_name:
        # Use folder name (training run) as agent name
        checkpoint_name = Path(checkpoint_path).parent.name
    else:
        # Use file name as agent name
        checkpoint_name = Path(checkpoint_path).stem
    
    if model_type == "dqn":
        # Auto-detect network_type from checkpoint
        network_type = DQNAgent.get_network_type_from_checkpoint(checkpoint_path)
        agent = DQNAgent(rows=6, cols=7, device=device, seed=seed, network_type=network_type)
        agent.load(checkpoint_path)
        agent.eval()
        agent.epsilon = epsilon
    elif model_type == "qlearning":
        agent = QLearningAgent(seed=seed)
        agent.load(checkpoint_path)
        agent.eval()
        agent.epsilon = epsilon
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return agent, checkpoint_name


def create_heuristic_agent(agent_type: str, seed: int = 42) -> Tuple[RandomAgent | HeuristicAgent | SmartHeuristicAgent, str]:
    """
    Create a heuristic agent.
    
    Args:
        agent_type: Type of heuristic ('random', 'heuristic', or 'smart_heuristic')
        seed: Random seed
        
    Returns:
        Tuple of (agent, agent_name)
    """
    if agent_type == "random":
        return RandomAgent(seed=seed), "Random"
    elif agent_type == "heuristic":
        return HeuristicAgent(seed=seed), "Heuristic"
    elif agent_type == "smart_heuristic":
        return SmartHeuristicAgent(seed=seed), "SmartHeuristic"
    else:
        raise ValueError(f"Unknown heuristic type: {agent_type}")


def tournament_all_models(
    checkpoint_dir: str = "data/checkpoints",
    model_type: Optional[Literal["dqn", "qlearning"]] = None,
    num_games_per_match: int = 10,
    include_heuristics: bool = True,
    heuristic_types: Optional[List[str]] = None,
    device: Optional[str] = None,
    seed: int = 42,
    epsilon: float = 0.0,
    reward_config: Optional[RewardConfig] = None,
    random_opening_config: Optional[RandomOpeningConfig] = None,
    verbose: bool = True,
) -> Dict:
    """
    Run a tournament between all trained models and heuristic agents.
    
    Args:
        checkpoint_dir: Directory to search for checkpoints
        model_type: Filter by model type ('dqn' or 'qlearning'), None for both
        num_games_per_match: Number of games per match
        include_heuristics: Whether to include heuristic agents in tournament
        heuristic_types: List of heuristic types to include (default: all)
        device: Device to use for DQN ('cuda' or 'cpu')
        seed: Random seed
        epsilon: Epsilon value for models (default: 0.0 for pure exploitation)
        reward_config: Reward configuration
        random_opening_config: Optional configuration for randomized opening prologues
        verbose: Whether to print progress
        
    Returns:
        Dictionary with results containing:
            - 'matrix': Win rate matrix (pandas DataFrame)
            - 'wins': Win count matrix
            - 'draws': Draw count matrix
            - 'losses': Loss count matrix
            - 'participant_names': List of participant names
            - 'total_games': Total number of games played
    """
    if reward_config is None:
        reward_config = RewardConfig()
    
    if heuristic_types is None:
        heuristic_types = ["random", "heuristic", "smart_heuristic"]
    
    # Find all checkpoints
    if verbose:
        print(f"Searching for checkpoints in {checkpoint_dir}...")
    
    checkpoint_paths = find_all_checkpoints(checkpoint_dir, model_type)
    
    if verbose:
        print(f"Found {len(checkpoint_paths)} checkpoints")
    
    # Load all agents
    agents = []
    participant_names = []
    
    # Load trained models
    for checkpoint_path in checkpoint_paths:
        # Determine model type from file extension
        if checkpoint_path.endswith(".pt"):
            cp_model_type = "dqn"
        elif checkpoint_path.endswith(".pkl"):
            cp_model_type = "qlearning"
        else:
            continue
        
        # Skip if model_type filter is set and doesn't match
        if model_type is not None and cp_model_type != model_type:
            continue
        
        try:
            agent, name = load_agent_from_checkpoint(
                checkpoint_path, cp_model_type, device, seed, epsilon
            )
            agents.append(agent)
            participant_names.append(name)
            if verbose:
                print(f"  ✓ Loaded: {name} ({cp_model_type})")
        except Exception as e:
            if verbose:
                print(f"  ✗ Failed to load {checkpoint_path}: {e}")
            continue
    
    # Add heuristic agents
    if include_heuristics:
        for heuristic_type in heuristic_types:
            agent, name = create_heuristic_agent(heuristic_type, seed)
            agents.append(agent)
            participant_names.append(name)
            if verbose:
                print(f"  ✓ Added: {name}")
    
    if len(agents) < 2:
        raise ValueError(f"Need at least 2 participants, found {len(agents)}")
    
    if verbose:
        print(f"\nTotal participants: {len(agents)}")
        print(f"Playing {num_games_per_match} games per match")
        if random_opening_config is not None:
            print(
                "Random opening prologue enabled: "
                f"p={random_opening_config.probability:.2f}, "
                f"half-moves={random_opening_config.min_half_moves}-{random_opening_config.max_half_moves}"
            )
        print()
    
    # Initialize result matrices
    n = len(agents)
    wins_matrix = np.zeros((n, n), dtype=int)
    draws_matrix = np.zeros((n, n), dtype=int)
    losses_matrix = np.zeros((n, n), dtype=int)
    
    total_matches = n * (n - 1)  # Each pair plays twice: once with each agent going first
    match_count = 0
    
    if verbose:
        print(f"Playing {total_matches} matches ({num_games_per_match} games each)...")
        print(f"Each pair plays twice: once with each agent going first")
        print("=" * 80)
    
    # Round-robin tournament
    for i in range(n):
        for j in range(i + 1, n):
            # Match 1: i first (player 1)
            match_count += 1
            if verbose:
                print(f"Match {match_count}/{total_matches}: {participant_names[i]} (first) vs {participant_names[j]}", end=" ... ")
            
            i_wins_first, draws, j_wins_first = play_match(
                agents[i],
                agents[j],
                num_games=num_games_per_match,
                seed=seed + match_count,
                randomize_first_player=False,  # Fixed order: agent1 (i) always goes first
                reward_config=reward_config,
                random_opening_config=random_opening_config,
            )
            
            wins_matrix[i, j] += i_wins_first
            wins_matrix[j, i] += j_wins_first
            draws_matrix[i, j] += draws
            draws_matrix[j, i] += draws
            
            if verbose:
                win_rate = i_wins_first / num_games_per_match
                print(f"Win rate: {win_rate:.1%} ({i_wins_first}/{num_games_per_match})")
            
            # Match 2: j first (player 1)
            match_count += 1
            if verbose:
                print(f"Match {match_count}/{total_matches}: {participant_names[j]} (first) vs {participant_names[i]}", end=" ... ")
            
            j_wins_second, draws_swap, i_wins_second = play_match(
                agents[j],
                agents[i],
                num_games=num_games_per_match,
                seed=seed + match_count,
                randomize_first_player=False,  # Fixed order: agent1 (j) always goes first
                reward_config=reward_config,
                random_opening_config=random_opening_config,
            )
            
            wins_matrix[i, j] += i_wins_second
            wins_matrix[j, i] += j_wins_second
            draws_matrix[i, j] += draws_swap
            draws_matrix[j, i] += draws_swap
            
            if verbose:
                win_rate = i_wins_second / num_games_per_match
                print(f"Win rate: {win_rate:.1%} ({i_wins_second}/{num_games_per_match})")
    
    # Calculate losses matrix: losses[i, j] = wins[j, i]
    losses_matrix = wins_matrix.T.copy()
    
    # Calculate win rate matrix
    # Each pair played 2 matches (one with each agent going first), so total games per pair = 2 * num_games_per_match
    win_rate_matrix = wins_matrix / (2 * num_games_per_match)
    
    # Create DataFrames
    wins_df = pd.DataFrame(wins_matrix, index=participant_names, columns=participant_names)
    draws_df = pd.DataFrame(draws_matrix, index=participant_names, columns=participant_names)
    losses_df = pd.DataFrame(losses_matrix, index=participant_names, columns=participant_names)
    win_rate_df = pd.DataFrame(win_rate_matrix, index=participant_names, columns=participant_names)
    
    # Calculate overall statistics
    total_wins = wins_matrix.sum(axis=1)
    total_draws = draws_matrix.sum(axis=1)
    total_losses = losses_matrix.sum(axis=1)
    total_games_played = total_wins + total_draws + total_losses
    overall_win_rate = total_wins / total_games_played if total_games_played.sum() > 0 else np.zeros(n)
    
    results = {
        'matrix': win_rate_df,
        'wins': wins_df,
        'draws': draws_df,
        'losses': losses_df,
        'participant_names': participant_names,
        'total_wins': total_wins,
        'total_draws': total_draws,
        'total_losses': total_losses,
        'overall_win_rate': overall_win_rate,
        'total_games': total_games_played,
        'num_games_per_match': num_games_per_match,
    }
    
    return results


def print_tournament_results(results: Dict):
    """Print tournament results in a readable format."""
    win_rate_df = results['matrix']
    participant_names = results['participant_names']
    total_wins = results['total_wins']
    total_draws = results['total_draws']
    total_losses = results['total_losses']
    overall_win_rate = results['overall_win_rate']
    total_games = results['total_games']
    
    print("\n" + "=" * 80)
    print("TOURNAMENT RESULTS")
    print("=" * 80)
    
    # Overall statistics
    print("\nOverall Statistics:")
    print(f"{'Participant':<30} {'Wins':<8} {'Draws':<8} {'Losses':<8} {'Win Rate':<10}")
    print("-" * 80)
    
    # Sort by win rate
    sorted_indices = np.argsort(overall_win_rate)[::-1]
    for idx in sorted_indices:
        name = participant_names[idx]
        wins = total_wins[idx]
        draws = total_draws[idx]
        losses = total_losses[idx]
        win_rate = overall_win_rate[idx]
        print(f"{name:<30} {wins:<8} {draws:<8} {losses:<8} {win_rate:>8.2%}")
    
    print("\n" + "=" * 80)
    print("Win Rate Matrix (Row vs Column):")
    print("=" * 80)
    print(win_rate_df.to_string())


def plot_tournament_results(
    results: Dict,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None,
):
    """
    Plot tournament results as a heatmap and bar chart.
    
    Args:
        results: Results dictionary from tournament_all_models
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    win_rate_df = results['matrix']
    participant_names = results['participant_names']
    overall_win_rate = results['overall_win_rate']
    total_wins = results['total_wins']
    total_losses = results['total_losses']
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Win rate heatmap
    if HAS_SEABORN:
        sns.heatmap(
            win_rate_df,
            annot=True,
            fmt='.2%',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            center=0.5,
            square=True,
            linewidths=0.5,
            cbar_kws={"label": "Win Rate"},
            ax=axes[0],
        )
    else:
        # Fallback to matplotlib imshow
        im = axes[0].imshow(win_rate_df.values, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        axes[0].set_xticks(range(len(win_rate_df.columns)))
        axes[0].set_yticks(range(len(win_rate_df.index)))
        axes[0].set_xticklabels(win_rate_df.columns, rotation=45, ha='right')
        axes[0].set_yticklabels(win_rate_df.index)
        # Add text annotations
        for i in range(len(win_rate_df.index)):
            for j in range(len(win_rate_df.columns)):
                text = axes[0].text(j, i, f'{win_rate_df.iloc[i, j]:.1%}',
                                  ha="center", va="center", color="black", fontsize=8)
        plt.colorbar(im, ax=axes[0], label="Win Rate")
    axes[0].set_title('Win Rate Matrix\n(Row vs Column)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Opponent (Column)', fontsize=12)
    axes[0].set_ylabel('Player (Row)', fontsize=12)
    
    # Overall statistics bar plot
    x_pos = np.arange(len(participant_names))
    width = 0.35
    
    # Sort by win rate
    sorted_indices = np.argsort(overall_win_rate)[::-1]
    sorted_names = [participant_names[i] for i in sorted_indices]
    sorted_wins = [total_wins[i] for i in sorted_indices]
    sorted_losses = [total_losses[i] for i in sorted_indices]
    
    axes[1].bar(x_pos - width/2, sorted_wins, width, label='Wins', color='green', alpha=0.7)
    axes[1].bar(x_pos + width/2, sorted_losses, width, label='Losses', color='red', alpha=0.7)
    axes[1].set_xlabel('Participant', fontsize=12)
    axes[1].set_ylabel('Number of Games', fontsize=12)
    axes[1].set_title('Overall Win/Loss Statistics', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(sorted_names, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()


def main(
    checkpoint_dir: str = "data/checkpoints",
    model_type: Optional[Literal["dqn", "qlearning"]] = None,
    num_games_per_match: int = 10,
    include_heuristics: bool = True,
    heuristic_types: Optional[List[str]] = None,
    device: Optional[str] = None,
    seed: int = 42,
    epsilon: float = 0.0,
    random_opening_config: Optional[RandomOpeningConfig] = None,
    verbose: bool = True,
    plot: bool = True,
    save_plot: Optional[str] = None,
):
    """
    Main function to run tournament.
    
    Args:
        checkpoint_dir: Directory to search for checkpoints
        model_type: Filter by model type ('dqn' or 'qlearning'), None for both
        num_games_per_match: Number of games per match
        include_heuristics: Whether to include heuristic agents
        heuristic_types: List of heuristic types to include
        device: Device to use for DQN
        seed: Random seed
        epsilon: Epsilon value for models
        random_opening_config: Optional configuration for randomized opening prologues
        verbose: Whether to print progress
        plot: Whether to plot results
        save_plot: Path to save plot (optional)
    """
    if heuristic_types is None:
        heuristic_types = ["random", "heuristic", "smart_heuristic"]
    
    results = tournament_all_models(
        checkpoint_dir=checkpoint_dir,
        model_type=model_type,
        num_games_per_match=num_games_per_match,
        include_heuristics=include_heuristics,
        heuristic_types=heuristic_types,
        device=device,
        seed=seed,
        epsilon=epsilon,
        random_opening_config=random_opening_config,
        verbose=verbose,
    )
    
    print_tournament_results(results)
    
    if plot:
        plot_tournament_results(results, save_path=save_plot)
    
    return results


if __name__ == "__main__":
    tyro.cli(main)

