"""Compare different agents against each other."""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.envs import Connect4Env
from src.agents import RandomAgent, HeuristicAgent, SmartHeuristicAgent


def play_match(agent1, agent2, num_games: int = 100, seed: int = 42) -> Tuple[int, int, int]:
    """
    Play matches between two agents.
    
    Args:
        agent1: First agent (plays as player 1)
        agent2: Second agent (plays as player -1)
        num_games: Number of games to play
        seed: Random seed
        
    Returns:
        Tuple of (agent1_wins, draws, agent2_wins)
    """
    env = Connect4Env(rows=6, cols=7)
    
    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    
    for game in range(num_games):
        obs = env.reset()
        done = False
        current_player = 0  # 0: agent1, 1: agent2
        
        while not done:
            legal_actions = env.get_legal_actions()
            
            if current_player == 0:
                action = agent1.select_action(obs, legal_actions)
            else:
                action = agent2.select_action(obs, legal_actions)
            
            next_obs, reward, done, info = env.step(action)
            
            if done:
                winner = info.get("winner")
                if winner == 1:  # Agent1 won
                    agent1_wins += 1
                elif winner == -1:  # Agent2 won
                    agent2_wins += 1
                else:  # Draw
                    draws += 1
                break
            
            obs = next_obs
            current_player = 1 - current_player
    
    return agent1_wins, draws, agent2_wins


def compare_agents(num_games: int = 1000, seed: int = 42):
    """
    Compare all agents against each other.
    
    Args:
        num_games: Number of games per match
        seed: Random seed
    """
    agents = {
        "random": RandomAgent(seed=seed),
        "heuristic": HeuristicAgent(seed=seed + 1),
        "smart_heuristic": SmartHeuristicAgent(seed=seed + 2),
    }
    
    agent_names = list(agents.keys())
    results = {}
    
    print("=" * 80)
    print("Agent Comparison - Win Rate Analysis")
    print("=" * 80)
    print(f"Games per match: {num_games}")
    print()
    
    # Play all combinations
    for i, name1 in enumerate(agent_names):
        for j, name2 in enumerate(agent_names):
            if i == j:
                continue
            
            print(f"Playing {name1} vs {name2}...", end=" ", flush=True)
            
            agent1 = agents[name1]
            agent2 = agents[name2]
            
            wins1, draws, wins2 = play_match(agent1, agent2, num_games, seed=seed + i * 10 + j)
            
            win_rate1 = wins1 / num_games
            draw_rate = draws / num_games
            win_rate2 = wins2 / num_games
            
            results[(name1, name2)] = {
                "wins1": wins1,
                "draws": draws,
                "wins2": wins2,
                "win_rate1": win_rate1,
                "draw_rate": draw_rate,
                "win_rate2": win_rate2,
            }
            
            print(f"Done! {name1}: {win_rate1:.1%} | Draws: {draw_rate:.1%} | {name2}: {win_rate2:.1%}")
    
    print()
    print("=" * 80)
    print("Results Summary")
    print("=" * 80)
    print()
    
    # Print results table
    print(f"{'Agent 1':<20} {'Agent 2':<20} {'Win Rate 1':<15} {'Draw Rate':<15} {'Win Rate 2':<15}")
    print("-" * 80)
    
    for (name1, name2), stats in results.items():
        print(
            f"{name1:<20} {name2:<20} "
            f"{stats['win_rate1']:>14.1%} {stats['draw_rate']:>14.1%} {stats['win_rate2']:>14.1%}"
        )
    
    print()
    print("=" * 80)
    print("Detailed Statistics")
    print("=" * 80)
    print()
    
    for (name1, name2), stats in results.items():
        print(f"{name1} vs {name2}:")
        print(f"  {name1} wins: {stats['wins1']} ({stats['win_rate1']:.1%})")
        print(f"  Draws: {stats['draws']} ({stats['draw_rate']:>6.1%})")
        print(f"  {name2} wins: {stats['wins2']} ({stats['win_rate2']:.1%})")
        print()
    
    # Calculate overall win rates
    print("=" * 80)
    print("Overall Win Rates (as Player 1)")
    print("=" * 80)
    print()
    
    overall_stats = {}
    for name in agent_names:
        wins = 0
        total = 0
        for (name1, name2), stats in results.items():
            if name1 == name:
                wins += stats['wins1']
                total += num_games
            elif name2 == name:
                wins += stats['wins2']
                total += num_games
        
        if total > 0:
            overall_win_rate = wins / total
            overall_stats[name] = overall_win_rate
            print(f"{name}: {overall_win_rate:.1%} ({wins}/{total})")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare different agents")
    parser.add_argument("--num-games", type=int, default=1000, help="Number of games per match")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    compare_agents(num_games=args.num_games, seed=args.seed)

