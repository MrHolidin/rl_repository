"""Compare multiple checkpoints against each other in a round-robin tournament."""

import sys
from pathlib import Path
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

from src.envs import Connect4Env, RewardConfig
from src.agents import DQNAgent, QLearningAgent
from src.features.action_space import DiscreteActionSpace
from src.features.observation_builder import BoardChannels
from src.utils.match import play_match


def load_agent_from_checkpoint(
    checkpoint_path: str,
    model_type: str,
    device: Optional[str] = None,
    seed: int = 42,
    epsilon: float = 0.01,
    use_epsilon: bool = False,
) -> Tuple[DQNAgent | QLearningAgent, str]:
    """
    Load an agent from a checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_type: Type of model ('dqn' or 'qlearning')
        device: Device to use for DQN ('cuda' or 'cpu')
        seed: Random seed
        epsilon: Epsilon value for epsilon-greedy exploration (used only if use_epsilon=True)
        use_epsilon: Whether to use epsilon-greedy exploration (default: False = pure exploitation)
        
    Returns:
        Tuple of (agent, checkpoint_name)
    """
    checkpoint_name = Path(checkpoint_path).stem
    
    if model_type == "dqn":
        # Auto-detect network_type from checkpoint
        observation_builder = BoardChannels(board_shape=(6, 7))
        action_space = DiscreteActionSpace(n=7)
        agent = DQNAgent.load(
            checkpoint_path,
            device=device,
            seed=seed,
            observation_shape=observation_builder.observation_shape,
            observation_type=observation_builder.observation_type,
            num_actions=action_space.size,
            action_space=action_space,
        )
        agent.eval()
        agent.epsilon = epsilon if use_epsilon else 0.0
    elif model_type == "qlearning":
        agent = QLearningAgent.load(checkpoint_path, seed=seed)
        agent.eval()
        agent.epsilon = epsilon if use_epsilon else 0.0
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return agent, checkpoint_name


def compare_checkpoints(
    checkpoint_paths: List[str],
    model_type: Literal["dqn", "qlearning"] = "dqn",
    num_games_per_match: int = 100,
    device: Optional[str] = None,
    seed: int = 42,
    epsilon: float = 0.01,
    use_epsilon: bool = False,
    reward_config: Optional[RewardConfig] = None,
    verbose: bool = True,
) -> Dict:
    """
    Compare multiple checkpoints against each other in a round-robin tournament.
    
    Args:
        checkpoint_paths: List of paths to checkpoint files
        model_type: Type of model ('dqn' or 'qlearning')
        num_games_per_match: Number of games per match
        device: Device to use for DQN ('cuda' or 'cpu')
        seed: Random seed
        epsilon: Epsilon value for epsilon-greedy exploration (used only if use_epsilon=True)
        use_epsilon: Whether to use epsilon-greedy exploration (default: False = pure exploitation)
        reward_*: Reward configuration
        verbose: Whether to print progress
        
    Returns:
        Dictionary with results containing:
            - 'matrix': Win rate matrix (pandas DataFrame)
            - 'wins': Win count matrix
            - 'draws': Draw count matrix
            - 'losses': Loss count matrix
            - 'checkpoint_names': List of checkpoint names
            - 'total_games': Total number of games played
    """
    if len(checkpoint_paths) < 2:
        raise ValueError("Need at least 2 checkpoints to compare")
    
    # Initialize reward config
    if reward_config is None:
        reward_config = RewardConfig()
    
    # Load all agents
    agents = []
    checkpoint_names = []
    
    if verbose:
        print(f"Loading {len(checkpoint_paths)} checkpoints...")
    
    for checkpoint_path in checkpoint_paths:
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        agent, name = load_agent_from_checkpoint(checkpoint_path, model_type, device, seed, epsilon, use_epsilon)
        agents.append(agent)
        checkpoint_names.append(name)
        
        if verbose:
            if use_epsilon:
                print(f"  ✓ Loaded: {name} (epsilon={epsilon:.2%})")
            else:
                print(f"  ✓ Loaded: {name} (pure exploitation, epsilon=0.0)")
    
    # Initialize result matrices
    n = len(agents)
    wins_matrix = np.zeros((n, n), dtype=int)
    draws_matrix = np.zeros((n, n), dtype=int)
    losses_matrix = np.zeros((n, n), dtype=int)
    
    total_matches = n * (n - 1)  # Each pair plays twice: once with each agent going first
    match_count = 0
    
    if verbose:
        print(f"\nPlaying {total_matches} matches ({num_games_per_match} games each)...")
        print(f"Each pair plays twice: once with each agent going first")
        print("=" * 80)
    
    # Round-robin tournament
    # Each pair plays twice: once with each agent going first
    # Iterate only over upper triangle (i < j) to avoid duplicate pairs
    for i in range(n):
        for j in range(i + 1, n):
            
            # Match 1: i first (player 1)
            match_count += 1
            if verbose:
                print(f"Match {match_count}/{total_matches}: {checkpoint_names[i]} (first) vs {checkpoint_names[j]}", end=" ... ")
            
            i_wins_first, draws, j_wins_first = play_match(
                agents[i],
                agents[j],
                num_games=num_games_per_match,
                seed=seed + match_count,
                randomize_first_player=False,  # Fixed order: agent1 (i) always goes first
                reward_config=reward_config,
            )
            
            wins_matrix[i, j] += i_wins_first   # победы i
            wins_matrix[j, i] += j_wins_first   # победы j
            draws_matrix[i, j] += draws
            draws_matrix[j, i] += draws
            
            if verbose:
                win_rate = i_wins_first / num_games_per_match
                print(f"Win rate: {win_rate:.1%} ({i_wins_first}/{num_games_per_match})")
            
            # Match 2: j first (player 1)
            match_count += 1
            if verbose:
                print(f"Match {match_count}/{total_matches}: {checkpoint_names[j]} (first) vs {checkpoint_names[i]}", end=" ... ")
            
            j_wins_second, draws_swap, i_wins_second = play_match(
                agents[j],
                agents[i],
                num_games=num_games_per_match,
                seed=seed + match_count,
                randomize_first_player=False,  # Fixed order: agent1 (j) always goes first
                reward_config=reward_config,
            )
            
            wins_matrix[i, j] += i_wins_second   # победы i, теперь он второй
            wins_matrix[j, i] += j_wins_second   # победы j, теперь он первый
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
    wins_df = pd.DataFrame(wins_matrix, index=checkpoint_names, columns=checkpoint_names)
    draws_df = pd.DataFrame(draws_matrix, index=checkpoint_names, columns=checkpoint_names)
    losses_df = pd.DataFrame(losses_matrix, index=checkpoint_names, columns=checkpoint_names)
    win_rate_df = pd.DataFrame(win_rate_matrix, index=checkpoint_names, columns=checkpoint_names)
    
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
        'checkpoint_names': checkpoint_names,
        'checkpoint_paths': checkpoint_paths,
        'total_wins': total_wins,
        'total_draws': total_draws,
        'total_losses': total_losses,
        'overall_win_rate': overall_win_rate,
        'total_games': total_games_played,
        'num_games_per_match': num_games_per_match,
    }
    
    return results


def plot_comparison_results(
    results: Dict,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
):
    """
    Plot comparison results as a heatmap.
    
    Args:
        results: Results dictionary from compare_checkpoints
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    win_rate_df = results['matrix']
    checkpoint_names = results['checkpoint_names']
    
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
    overall_win_rate = results['overall_win_rate']
    total_wins = results['total_wins']
    total_losses = results['total_losses']
    
    x_pos = np.arange(len(checkpoint_names))
    width = 0.35
    
    axes[1].bar(x_pos - width/2, total_wins, width, label='Wins', color='green', alpha=0.7)
    axes[1].bar(x_pos + width/2, total_losses, width, label='Losses', color='red', alpha=0.7)
    axes[1].set_xlabel('Checkpoint', fontsize=12)
    axes[1].set_ylabel('Number of Games', fontsize=12)
    axes[1].set_title('Overall Win/Loss Statistics', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(checkpoint_names, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add win rate annotations
    for i, (name, win_rate) in enumerate(zip(checkpoint_names, overall_win_rate)):
        axes[1].text(i, max(total_wins[i], total_losses[i]) + 5, f'{win_rate:.1%}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def print_comparison_results(results: Dict):
    """
    Print comparison results in a readable format.
    
    Args:
        results: Results dictionary from compare_checkpoints
    """
    win_rate_df = results['matrix']
    checkpoint_names = results['checkpoint_names']
    overall_win_rate = results['overall_win_rate']
    total_wins = results['total_wins']
    total_draws = results['total_draws']
    total_losses = results['total_losses']
    
    print("\n" + "=" * 80)
    print("CHECKPOINT COMPARISON RESULTS")
    print("=" * 80)
    print(f"\nWin Rate Matrix (Row vs Column):")
    print(win_rate_df.to_string())
    
    print(f"\n{'=' * 80}")
    print("Overall Statistics:")
    print(f"{'=' * 80}")
    print(f"{'Checkpoint':<40} {'Win Rate':<12} {'Wins':<8} {'Draws':<8} {'Losses':<8}")
    print("-" * 80)
    
    # Sort by win rate
    sorted_indices = np.argsort(overall_win_rate)[::-1]
    
    for idx in sorted_indices:
        name = checkpoint_names[idx]
        win_rate = overall_win_rate[idx]
        wins = total_wins[idx]
        draws = total_draws[idx]
        losses = total_losses[idx]
        print(f"{name:<40} {win_rate:>10.1%}  {wins:>6}  {draws:>6}  {losses:>6}")
    
    print("=" * 80)


def main(
    checkpoints: list[str],
    model_type: Literal["dqn", "qlearning"] = "dqn",
    num_games: int = 100,
    device: Optional[str] = None,
    seed: int = 42,
    epsilon: float = 0.01,
    use_epsilon: bool = False,
    save_plot: Optional[str] = None,
):
    """Main entry point for checkpoint comparison."""
    results = compare_checkpoints(
        checkpoint_paths=checkpoints,
        model_type=model_type,
        num_games_per_match=num_games,
        device=device,
        seed=seed,
        epsilon=epsilon,
        use_epsilon=use_epsilon,
    )
    
    print_comparison_results(results)
    plot_comparison_results(results, save_path=save_plot)


if __name__ == "__main__":
    import tyro
    tyro.cli(main)

