"""Training script for DQN agent."""

import argparse
import os
import sys
import time
import threading
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.envs import Connect4Env
from src.agents import DQNAgent, RandomAgent, HeuristicAgent, SmartHeuristicAgent
from src.utils import MetricsLogger


def train_dqn(
    num_episodes: int = 10000,
    learning_rate: float = 0.001,
    discount_factor: float = 0.99,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.995,
    epsilon_min: float = 0.01,
    batch_size: int = 32,
    replay_buffer_size: int = 10000,
    target_update_freq: int = 100,
    soft_update: bool = False,
    tau: float = 0.01,
    opponent_type: str = "random",
    eval_freq: int = 100,
    eval_episodes: int = 100,
    save_freq: int = 1000,
    checkpoint_dir: str = "data/checkpoints",
    log_dir: str = "data/logs",
    device: str = None,
    seed: int = 42,
    stop_flag: threading.Event = None,
    reward_win: float = 1.0,
    reward_loss: float = -1.0,
    reward_draw: float = 0.0,
    reward_three_in_row: float = 0.0,
    reward_invalid_action: float = -0.1,
):
    """
    Train DQN agent using self-play.
    
    Args:
        num_episodes: Number of training episodes
        learning_rate: Learning rate
        discount_factor: Discount factor
        epsilon: Initial epsilon
        epsilon_decay: Epsilon decay rate
        epsilon_min: Minimum epsilon
        batch_size: Batch size for training
        replay_buffer_size: Size of replay buffer
        target_update_freq: Target network update frequency
        opponent_type: Type of opponent ('random', 'heuristic', or 'smart_heuristic')
        eval_freq: Evaluation frequency
        eval_episodes: Number of episodes for evaluation
        save_freq: Checkpoint save frequency
        checkpoint_dir: Directory for checkpoints
        log_dir: Directory for logs
        device: Device to use ('cuda' or 'cpu')
        seed: Random seed
    """
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize environment with reward configuration
    env = Connect4Env(
        rows=6, 
        cols=7,
        reward_win=reward_win,
        reward_loss=reward_loss,
        reward_draw=reward_draw,
        reward_three_in_row=reward_three_in_row,
        reward_invalid_action=reward_invalid_action,
    )
    
    # Initialize agents
    learning_agent = DQNAgent(
        rows=6,
        cols=7,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        batch_size=batch_size,
        replay_buffer_size=replay_buffer_size,
        target_update_freq=target_update_freq,
        soft_update=soft_update,
        tau=tau,
        device=device,
        seed=seed,
    )
    
    # Create opponent based on type
    if opponent_type == "random":
        opponent = RandomAgent(seed=seed + 1)
    elif opponent_type == "heuristic":
        opponent = HeuristicAgent(seed=seed + 1)
    elif opponent_type == "smart_heuristic":
        opponent = SmartHeuristicAgent(seed=seed + 1)
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}. Must be 'random', 'heuristic', or 'smart_heuristic'")
    
    # Initialize logger
    logger = MetricsLogger(log_dir=log_dir)
    
    print("Starting DQN training...")
    print(f"Episodes: {num_episodes}")
    print(f"Learning rate: {learning_rate}")
    print(f"Discount factor: {discount_factor}")
    print(f"Initial epsilon: {epsilon}")
    print(f"Target update: {'soft' if soft_update else 'hard'} (freq: {target_update_freq})")
    if soft_update:
        print(f"Soft update tau: {tau}")
    print(f"Opponent: {opponent_type}")
    print(f"Device: {learning_agent.device}")
    print()
    
    # Track training metrics
    episode_start_time = time.time()
    training_metrics_accumulator = defaultdict(list)
    training_steps_count = 0
    
    training_completed = False
    for episode in range(num_episodes):
        # Check stop flag
        if stop_flag is not None and stop_flag.is_set():
            print(f"\n🛑 Обучение остановлено пользователем на эпизоде {episode + 1}/{num_episodes}")
            # Save checkpoint before stopping
            checkpoint_path = os.path.join(checkpoint_dir, f"dqn_episode_{episode + 1}_stopped.pt")
            learning_agent.save(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
            break
        
        episode_start = time.time()
        obs = env.reset()
        done = False
        current_player = 0  # 0: learning_agent, 1: opponent
        episode_reward = 0
        episode_length = 0
        episode_training_steps = 0
        
        # Store last observation from learning agent's perspective
        last_learning_obs = None
        last_learning_action = None
        
        while not done:
            legal_actions = env.get_legal_actions()
            
            if current_player == 0:
                # Learning agent's turn
                action = learning_agent.select_action(obs, legal_actions)
                last_learning_obs = obs.copy()  # Save observation before action
                last_learning_action = action
            else:
                # Opponent's turn
                action = opponent.select_action(obs, legal_actions)
            
            next_obs, reward, done, info = env.step(action)
            
            # Learning agent observes transition
            # Note: reward is from perspective of current player (who made the move)
            if current_player == 0:
                # Learning agent's perspective - reward is already correct
                train_metrics = learning_agent.observe((obs, action, reward, next_obs, done, info))
                episode_reward += reward
                
                # Accumulate training metrics
                if train_metrics:
                    episode_training_steps += 1
                    training_steps_count += 1
                    for key, value in train_metrics.items():
                        if key != "target_network_updated":  # Handle boolean separately
                            training_metrics_accumulator[key].append(value)
            else:
                # Opponent's move - if game ended, we need to give final reward to learning agent
                if done:
                    # Get final reward for learning agent from learning agent's perspective
                    winner = info.get("winner")
                    if winner == 1:  # Learning agent won (player 1)
                        final_reward = 1.0
                    elif winner == -1:  # Opponent won (player -1)
                        final_reward = -1.0
                    else:  # Draw
                        final_reward = 0.0
                    
                    # Observe final transition from learning agent's last state
                    if last_learning_obs is not None and last_learning_action is not None:
                        # Use the observation from learning agent's perspective before opponent's move
                        # The next_obs is from opponent's perspective, so we need to flip it
                        # Actually, next_obs is already from the perspective of the next player (opponent)
                        # We need to create an observation from learning agent's perspective
                        # For simplicity, we'll use next_obs but with inverted perspective
                        # Actually, the best approach is to use the observation as if it's learning agent's turn
                        # But since the game is done, we can use a dummy next_obs
                        final_next_obs = next_obs.copy()  # This is from opponent's perspective
                        # We need to flip it to learning agent's perspective
                        # The observation format is (3, rows, cols) where channels are:
                        # 0: current player's pieces, 1: opponent's pieces, 2: current player indicator
                        # We need to swap channels 0 and 1, and flip channel 2
                        final_next_obs_flipped = final_next_obs.copy()
                        final_next_obs_flipped[0] = next_obs[1]  # Current player becomes opponent
                        final_next_obs_flipped[1] = next_obs[0]  # Opponent becomes current player
                        final_next_obs_flipped[2] = 1.0 - next_obs[2]  # Flip player indicator
                        
                        train_metrics = learning_agent.observe((
                            last_learning_obs,
                            last_learning_action,
                            final_reward,
                            final_next_obs_flipped,
                            True,  # done
                            info
                        ))
                        episode_reward += final_reward
                        
                        # Accumulate training metrics
                        if train_metrics:
                            episode_training_steps += 1
                            training_steps_count += 1
                            for key, value in train_metrics.items():
                                if key != "target_network_updated":
                                    training_metrics_accumulator[key].append(value)
            
            obs = next_obs
            current_player = 1 - current_player
            episode_length += 1
        
        episode_time = time.time() - episode_start
        
        # Compute average training metrics for this episode
        episode_training_metrics = {}
        for key, values in training_metrics_accumulator.items():
            if values:
                episode_training_metrics[f"train_{key}"] = sum(values) / len(values)
                episode_training_metrics[f"train_{key}_total"] = sum(values)
        
        # Log basic episode metrics
        logger.log("episode_reward", episode_reward, step=episode)
        logger.log("episode_length", episode_length, step=episode)
        logger.log("episode_time", episode_time, step=episode)
        logger.log("epsilon", learning_agent.epsilon, step=episode)
        logger.log("replay_buffer_size", len(learning_agent.replay_buffer), step=episode)
        buffer_capacity = learning_agent.replay_buffer.capacity
        logger.log("replay_buffer_utilization", len(learning_agent.replay_buffer) / buffer_capacity if buffer_capacity > 0 else 0.0, step=episode)
        logger.log("training_steps_per_episode", episode_training_steps, step=episode)
        logger.log("total_training_steps", training_steps_count, step=episode)
        logger.log("step_count", learning_agent.step_count, step=episode)
        
        # Log training metrics (averaged over episode)
        if episode_training_metrics:
            logger.log_dict(episode_training_metrics, step=episode)
        
        # Clear accumulator for next episode
        training_metrics_accumulator.clear()
        
        # Decay epsilon once per episode (not per step!)
        if learning_agent.training and learning_agent.epsilon > learning_agent.epsilon_min:
            learning_agent.epsilon *= learning_agent.epsilon_decay
        
        logger.increment_episode()
        
        # Evaluation (можно уменьшить частоту для более быстрого обновления)
        if (episode + 1) % eval_freq == 0:
            win_rate, draw_rate, loss_rate = evaluate_agent(
                learning_agent, 
                opponent, 
                eval_episodes, 
                seed=seed,
                reward_win=reward_win,
                reward_loss=reward_loss,
                reward_draw=reward_draw,
                reward_three_in_row=reward_three_in_row,
                reward_invalid_action=reward_invalid_action,
            )
            logger.log_dict({
                "win_rate": win_rate,
                "draw_rate": draw_rate,
                "loss_rate": loss_rate,
            }, step=episode)
            
            # Get latest training metrics for display
            avg_loss = episode_training_metrics.get("train_loss", 0.0)
            avg_q = episode_training_metrics.get("train_avg_q", 0.0)
            avg_td_error = episode_training_metrics.get("train_td_error", 0.0)
            grad_norm = episode_training_metrics.get("train_grad_norm", 0.0)
            buffer_util = len(learning_agent.replay_buffer) / buffer_capacity * 100 if buffer_capacity > 0 else 0.0
            
            print(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Win: {win_rate:.2%} | Draw: {draw_rate:.2%} | Loss: {loss_rate:.2%} | "
                f"Epsilon: {learning_agent.epsilon:.4f} | "
                f"Buffer: {len(learning_agent.replay_buffer)}/{buffer_capacity} ({buffer_util:.1f}%) | "
                f"Train steps: {episode_training_steps} | "
                f"Loss: {avg_loss:.4f} | Avg Q: {avg_q:.4f} | TD Error: {avg_td_error:.4f} | Grad Norm: {grad_norm:.4f}"
            )
        
        # Periodic detailed metrics report
        if (episode + 1) % (eval_freq // 2) == 0 and episode > 0:
            # Get more detailed metrics
            max_q = episode_training_metrics.get("train_max_q", 0.0)
            min_q = episode_training_metrics.get("train_min_q", 0.0)
            avg_target_q = episode_training_metrics.get("train_avg_target_q", 0.0)
            grad_norm_clipped = episode_training_metrics.get("train_grad_norm_clipped", 0.0)
            
            print(
                f"  Detailed metrics: Max Q: {max_q:.4f} | Min Q: {min_q:.4f} | "
                f"Target Q: {avg_target_q:.4f} | Grad Norm (clipped): {grad_norm_clipped:.4f} | "
                f"Total train steps: {training_steps_count} | Step count: {learning_agent.step_count}"
            )
        
        # Save checkpoint
        if (episode + 1) % save_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"dqn_episode_{episode + 1}.pt")
            learning_agent.save(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Mark training as completed if we reached the end
    if episode == num_episodes - 1:
        training_completed = True
    
    # Final save - always save final checkpoint, even if stopped early
    final_path = os.path.join(checkpoint_dir, "dqn_final.pt")
    learning_agent.save(final_path)
    if training_completed:
        print(f"\nTraining completed! Final model saved: {final_path}")
    else:
        print(f"\nTraining stopped. Final model saved: {final_path}")
    
    logger.close()


def evaluate_agent(
    agent, 
    opponent, 
    num_episodes: int, 
    seed: int = None,
    reward_win: float = 1.0,
    reward_loss: float = -1.0,
    reward_draw: float = 0.0,
    reward_three_in_row: float = 0.0,
    reward_invalid_action: float = -0.1,
) -> tuple:
    """
    Evaluate agent against opponent.
    
    Args:
        agent: Agent to evaluate
        opponent: Opponent agent
        num_episodes: Number of evaluation episodes
        seed: Random seed
        
    Returns:
        Tuple of (win_rate, draw_rate, loss_rate)
    """
    env = Connect4Env(
        rows=6, 
        cols=7,
        reward_win=reward_win,
        reward_loss=reward_loss,
        reward_draw=reward_draw,
        reward_three_in_row=reward_three_in_row,
        reward_invalid_action=reward_invalid_action,
    )
    
    agent.eval()
    wins = 0
    draws = 0
    losses = 0
    
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        current_player = 0  # 0: agent, 1: opponent
        
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
    
    agent.train()
    
    win_rate = wins / num_episodes
    draw_rate = draws / num_episodes
    loss_rate = losses / num_episodes
    
    return win_rate, draw_rate, loss_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN agent")
    parser.add_argument("--num-episodes", type=int, default=10000, help="Number of training episodes")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--discount-factor", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Epsilon decay rate")
    parser.add_argument("--epsilon-min", type=float, default=0.01, help="Minimum epsilon")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--replay-buffer-size", type=int, default=10000, help="Replay buffer size")
    parser.add_argument("--target-update-freq", type=int, default=100, help="Target update frequency")
    parser.add_argument("--soft-update", action="store_true", help="Use soft target network update")
    parser.add_argument("--tau", type=float, default=0.01, help="Soft update coefficient (only used with --soft-update)")
    parser.add_argument("--opponent-type", type=str, choices=["random", "heuristic", "smart_heuristic"], default="random", help="Type of opponent agent")
    parser.add_argument("--eval-freq", type=int, default=100, help="Evaluation frequency")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Evaluation episodes")
    parser.add_argument("--save-freq", type=int, default=1000, help="Checkpoint save frequency")
    parser.add_argument("--checkpoint-dir", type=str, default="data/checkpoints", help="Checkpoint directory")
    parser.add_argument("--log-dir", type=str, default="data/logs", help="Log directory")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    train_dqn(
        num_episodes=args.num_episodes,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        batch_size=args.batch_size,
        replay_buffer_size=args.replay_buffer_size,
        target_update_freq=args.target_update_freq,
        soft_update=args.soft_update,
        tau=args.tau,
        opponent_type=args.opponent_type,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        save_freq=args.save_freq,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        device=args.device,
        seed=args.seed,
    )

