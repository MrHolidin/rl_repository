"""Training script for DQN agent."""

import argparse
import os
import sys
import time
import threading
import random
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.envs import Connect4Env
from src.agents import DQNAgent, RandomAgent, HeuristicAgent, SmartHeuristicAgent
from src.utils import MetricsLogger
from src.selfplay import OpponentPool


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
    reward_opponent_three_in_row: float = 0.0,
    reward_invalid_action: float = -0.1,
    self_play_config: dict = None,
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
        self_play_config: Self-play configuration dict (optional)
    """
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize self-play config
    if self_play_config is None:
        self_play_config = {
            "enabled": False,
            "start_episode": 0,
            "save_every": save_freq,
            "self_play_fraction": 0.0,
            "max_frozen_agents": 10,
            "heuristic_distribution": {
                "random": 0.2 if opponent_type == "random" else 0.0,
                "heuristic": 0.5 if opponent_type == "heuristic" else 0.0,
                "smart_heuristic": 0.3 if opponent_type == "smart_heuristic" else 0.0,
            }
        }
        if not self_play_config["enabled"]:
            self_play_config["heuristic_distribution"] = {
                "random": 1.0 if opponent_type == "random" else 0.0,
                "heuristic": 1.0 if opponent_type == "heuristic" else 0.0,
                "smart_heuristic": 1.0 if opponent_type == "smart_heuristic" else 0.0,
            }
    
    # Initialize opponent pool
    opponent_pool = OpponentPool(
        device=device,
        seed=seed,
        self_play_enabled=self_play_config.get("enabled", False),
        self_play_start_episode=self_play_config.get("start_episode", 0),
        self_play_fraction=self_play_config.get("self_play_fraction", 0.0),
        max_frozen_agents=self_play_config.get("max_frozen_agents", 10),
        heuristic_distribution=self_play_config.get("heuristic_distribution", {
            "random": 0.2,
            "heuristic": 0.5,
            "smart_heuristic": 0.3,
        }),
    )
    
    # Initialize environment with reward configuration
    env = Connect4Env(
        rows=6, 
        cols=7,
        reward_win=reward_win,
        reward_loss=reward_loss,
        reward_draw=reward_draw,
        reward_three_in_row=reward_three_in_row,
        reward_opponent_three_in_row=reward_opponent_three_in_row,
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
    print(f"Device: {learning_agent.device}")
    if self_play_config.get("enabled", False):
        print(f"Self-play: ENABLED")
        print(f"  Start episode: {self_play_config.get('start_episode', 0)}")
        print(f"  Self-play fraction: {self_play_config.get('self_play_fraction', 0.0):.1%}")
        print(f"  Save every: {self_play_config.get('save_every', save_freq)} episodes")
        print(f"  Max frozen agents: {self_play_config.get('max_frozen_agents', 10)}")
    else:
        print(f"Opponent: {opponent_type} (heuristic)")
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
        
        opponent = opponent_pool.sample_opponent(episode)
        
        episode_start = time.time()
        obs = env.reset()
        done = False
        learning_agent_goes_first = random.random() < 0.5
        current_player = 0 if learning_agent_goes_first else 1
        episode_reward = 0
        episode_length = 0
        episode_training_steps = 0
        
        # Pending transition: сохраняем переход агента до хода соперника
        pending_obs = None
        pending_action = None
        pending_reward = 0.0
        
        while not done:
            legal_actions = env.get_legal_actions()
            
            if current_player == 0:
                # Ход агента
                action = learning_agent.select_action(obs, legal_actions)
                next_obs, reward, done, info = env.step(action)
                
                if done:
                    # Агент выиграл или ничья после его хода
                    # Записываем transition сразу с финальным ревардом
                    final_reward = reward  # env уже дал reward_win или reward_draw
                    train_metrics = learning_agent.observe((
                        obs, action, final_reward, next_obs, True, info
                    ))
                    episode_reward += final_reward
                    
                    if train_metrics:
                        episode_training_steps += 1
                        training_steps_count += 1
                        for key, value in train_metrics.items():
                            if key != "target_network_updated":
                                training_metrics_accumulator[key].append(value)
                else:
                    # Игра продолжается - сохраняем переход до хода соперника
                    pending_obs = obs
                    pending_action = action
                    pending_reward = reward  # shaping reward (тройки и т.п.)
                    # Не записываем observe пока соперник не сходил
                
                obs = next_obs  # теперь POV соперника
            else:
                # Ход соперника
                action = opponent.select_action(obs, legal_actions)
                next_obs, opp_reward, done, info = env.step(action)
                # opp_reward - для соперника, нам не нужен
                
                if pending_obs is not None:
                    # Завершаем переход агента
                    if done:
                        # Соперник завершил игру
                        winner = info.get("winner")
                        if winner == 0:
                            # Ничья
                            final_reward = reward_draw
                        else:
                            # Соперник победил → агент проиграл
                            final_reward = reward_loss  # отрицательный ревард
                        
                        # next_obs уже POV агента (после хода соперника current_player снова = агент)
                        train_metrics = learning_agent.observe((
                            pending_obs, pending_action, final_reward, next_obs, True, info
                        ))
                        episode_reward += final_reward
                    else:
                        # Игра продолжается после хода соперника
                        # next_obs уже POV агента
                        final_reward = pending_reward  # shaping reward за ход агента
                        train_metrics = learning_agent.observe((
                            pending_obs, pending_action, final_reward, next_obs, False, info
                        ))
                        episode_reward += final_reward
                    
                    if train_metrics:
                        episode_training_steps += 1
                        training_steps_count += 1
                        for key, value in train_metrics.items():
                            if key != "target_network_updated":
                                training_metrics_accumulator[key].append(value)
                    
                    # Очищаем pending transition
                    pending_obs = None
                    pending_action = None
                    pending_reward = 0.0
                
                obs = next_obs  # POV агента
            
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
        
        training_metrics_accumulator.clear()
        
        if learning_agent.training and learning_agent.epsilon > learning_agent.epsilon_min:
            learning_agent.epsilon *= learning_agent.epsilon_decay
        
        logger.increment_episode()
        
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
                reward_opponent_three_in_row=reward_opponent_three_in_row,
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
        
        if (episode + 1) % (eval_freq // 2) == 0 and episode > 0:
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
            
            if self_play_config.get("enabled", False):
                save_every = self_play_config.get("save_every", save_freq)
                if (episode + 1) % save_every == 0:
                    opponent_pool.add_frozen_agent(checkpoint_path, episode + 1)
                    pool_stats = opponent_pool.get_pool_stats()
                    print(f"  Added frozen opponent from episode {episode + 1} to pool (total: {pool_stats['frozen_agents_count']})")
    
    final_episode = episode + 1 if 'episode' in locals() else num_episodes
    if 'episode' in locals() and episode == num_episodes - 1:
        training_completed = True
    else:
        training_completed = False
    
    final_path = os.path.join(checkpoint_dir, "dqn_final.pt")
    learning_agent.save(final_path)
    
    if self_play_config.get("enabled", False):
        opponent_pool.add_frozen_agent(final_path, final_episode)
        pool_stats = opponent_pool.get_pool_stats()
        print(f"  Added final frozen opponent to pool (total: {pool_stats['frozen_agents_count']})")
    
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
    reward_opponent_three_in_row: float = 0.0,
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
        reward_opponent_three_in_row=reward_opponent_three_in_row,
        reward_invalid_action=reward_invalid_action,
    )
    
    # Сохраняем старое значение epsilon и устанавливаем 0 для жадной политики при оценке
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    agent.eval()
    wins = 0
    draws = 0
    losses = 0
    
    if seed is not None:
        random.seed(seed)
    
    for episode_idx in range(num_episodes):
        obs = env.reset()
        done = False
        # Рандомизируем, кто ходит первым (как в тренировке)
        agent_goes_first = random.random() < 0.5
        agent_is_player_1 = agent_goes_first  # Если агент ходит первым, он player 1
        
        while not done:
            legal_actions = env.get_legal_actions()
            
            # Определяем, кто должен ходить на основе env.current_player
            # env.current_player == 1 означает player 1
            # env.current_player == -1 означает player -1
            if env.current_player == 1:
                # Ходит player 1
                if agent_is_player_1:
                    action = agent.select_action(obs, legal_actions)
                else:
                    action = opponent.select_action(obs, legal_actions)
            else:
                # Ходит player -1
                if agent_is_player_1:
                    action = opponent.select_action(obs, legal_actions)
                else:
                    action = agent.select_action(obs, legal_actions)
            
            next_obs, reward, done, info = env.step(action)
            
            if done:
                winner = info.get("winner")
                # winner == 1 означает player 1 выиграл
                # winner == -1 означает player -1 выиграл
                # winner == 0 означает ничью
                if winner == 0:
                    draws += 1
                elif (winner == 1 and agent_is_player_1) or (winner == -1 and not agent_is_player_1):
                    # Агент выиграл
                    wins += 1
                else:
                    # Соперник выиграл
                    losses += 1
                break
            
            obs = next_obs
    
    # Восстанавливаем старое значение epsilon и переводим агента обратно в режим обучения
    agent.epsilon = old_epsilon
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
    parser.add_argument("--reward-win", type=float, default=1.0, help="Reward for winning")
    parser.add_argument("--reward-loss", type=float, default=-1.0, help="Reward for losing")
    parser.add_argument("--reward-draw", type=float, default=0.0, help="Reward for draw")
    parser.add_argument("--reward-three-in-row", type=float, default=0.0, help="Reward for getting 3 in a row")
    parser.add_argument("--reward-opponent-three-in-row", type=float, default=0.0, help="Penalty for opponent having 3 in a row")
    parser.add_argument("--reward-invalid-action", type=float, default=-0.1, help="Reward for invalid action")
    
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
        reward_win=args.reward_win,
        reward_loss=args.reward_loss,
        reward_draw=args.reward_draw,
        reward_three_in_row=args.reward_three_in_row,
        reward_opponent_three_in_row=args.reward_opponent_three_in_row,
        reward_invalid_action=args.reward_invalid_action,
    )

