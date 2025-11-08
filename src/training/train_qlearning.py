"""Training script for Q-learning agent."""

import argparse
import os
import sys
import threading
import random
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.envs import Connect4Env
from src.agents import QLearningAgent, RandomAgent, HeuristicAgent, SmartHeuristicAgent
from src.utils import MetricsLogger


def train_qlearning(
    num_episodes: int = 10000,
    learning_rate: float = 0.1,
    discount_factor: float = 0.99,
    epsilon: float = 0.1,
    epsilon_decay: float = 0.995,
    epsilon_min: float = 0.01,
    opponent_type: str = "random",
    eval_freq: int = 100,
    eval_episodes: int = 100,
    save_freq: int = 1000,
    checkpoint_dir: str = "data/checkpoints",
    log_dir: str = "data/logs",
    seed: int = 42,
    stop_flag: threading.Event = None,
    reward_win: float = 1.0,
    reward_loss: float = -1.0,
    reward_draw: float = 0.0,
    reward_three_in_row: float = 0.0,
    reward_opponent_three_in_row: float = 0.0,
    reward_invalid_action: float = -0.1,
):
    """
    Train Q-learning agent using self-play.
    
    Args:
        num_episodes: Number of training episodes
        learning_rate: Learning rate
        discount_factor: Discount factor
        epsilon: Initial epsilon
        epsilon_decay: Epsilon decay rate
        epsilon_min: Minimum epsilon
        opponent_type: Type of opponent ('random', 'heuristic', or 'smart_heuristic')
        eval_freq: Evaluation frequency
        eval_episodes: Number of episodes for evaluation
        save_freq: Checkpoint save frequency
        checkpoint_dir: Directory for checkpoints
        log_dir: Directory for logs
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
        reward_opponent_three_in_row=reward_opponent_three_in_row,
        reward_invalid_action=reward_invalid_action,
    )
    
    # Initialize agents
    learning_agent = QLearningAgent(
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
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
    
    print("Starting Q-learning training...")
    print(f"Episodes: {num_episodes}")
    print(f"Learning rate: {learning_rate}")
    print(f"Discount factor: {discount_factor}")
    print(f"Initial epsilon: {epsilon}")
    print(f"Opponent: {opponent_type}")
    print()
    
    training_completed = False
    for episode in range(num_episodes):
        # Check stop flag
        if stop_flag is not None and stop_flag.is_set():
            print(f"\n🛑 Обучение остановлено пользователем на эпизоде {episode + 1}/{num_episodes}")
            # Save checkpoint before stopping
            checkpoint_path = os.path.join(checkpoint_dir, f"qlearning_episode_{episode + 1}_stopped.pkl")
            learning_agent.save(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
            break
        
        obs = env.reset()
        done = False
        current_player = 0  # 0: learning_agent, 1: opponent
        episode_reward = 0
        episode_length = 0
        
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
                    learning_agent.observe((obs, action, final_reward, next_obs, True, info))
                    episode_reward += final_reward
                else:
                    # Игра продолжается - сохраняем переход до хода соперника
                    pending_obs = obs
                    pending_action = action
                    pending_reward = reward  # shaping reward (тройки и т.п.)
                    # Не записываем observe пока соперник не сходил
                
                obs = next_obs
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
                        
                        learning_agent.observe((
                            pending_obs, pending_action, final_reward, next_obs, True, info
                        ))
                        episode_reward += final_reward
                    else:
                        # Игра продолжается после хода соперника
                        final_reward = pending_reward  # shaping reward за ход агента
                        learning_agent.observe((
                            pending_obs, pending_action, final_reward, next_obs, False, info
                        ))
                        episode_reward += final_reward
                    
                    # Очищаем pending transition
                    pending_obs = None
                    pending_action = None
                    pending_reward = 0.0
                
                obs = next_obs
            
            current_player = 1 - current_player
            episode_length += 1
        
        # Log metrics
        logger.log("episode_reward", episode_reward, step=episode)
        logger.log("episode_length", episode_length, step=episode)
        logger.log("epsilon", learning_agent.epsilon, step=episode)
        
        # Decay epsilon once per episode (not per step!)
        if learning_agent.training and learning_agent.epsilon > learning_agent.epsilon_min:
            learning_agent.epsilon *= learning_agent.epsilon_decay
        
        logger.increment_episode()
        
        # Evaluation
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
            
            print(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Win rate: {win_rate:.2%} | "
                f"Draw rate: {draw_rate:.2%} | "
                f"Loss rate: {loss_rate:.2%} | "
                f"Epsilon: {learning_agent.epsilon:.4f}"
            )
        
        # Save checkpoint
        if (episode + 1) % save_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"qlearning_episode_{episode + 1}.pkl")
            learning_agent.save(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Mark training as completed if we reached the end
    if episode == num_episodes - 1:
        training_completed = True
    
    # Final save - always save final checkpoint, even if stopped early
    final_path = os.path.join(checkpoint_dir, "qlearning_final.pkl")
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
    parser = argparse.ArgumentParser(description="Train Q-learning agent")
    parser.add_argument("--num-episodes", type=int, default=10000, help="Number of training episodes")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--discount-factor", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Initial epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Epsilon decay rate")
    parser.add_argument("--epsilon-min", type=float, default=0.01, help="Minimum epsilon")
    parser.add_argument("--opponent-type", type=str, choices=["random", "heuristic", "smart_heuristic"], default="random", help="Type of opponent agent")
    parser.add_argument("--eval-freq", type=int, default=100, help="Evaluation frequency")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Evaluation episodes")
    parser.add_argument("--save-freq", type=int, default=1000, help="Checkpoint save frequency")
    parser.add_argument("--checkpoint-dir", type=str, default="data/checkpoints", help="Checkpoint directory")
    parser.add_argument("--log-dir", type=str, default="data/logs", help="Log directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    train_qlearning(
        num_episodes=args.num_episodes,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        opponent_type=args.opponent_type,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        save_freq=args.save_freq,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        seed=args.seed,
    )

