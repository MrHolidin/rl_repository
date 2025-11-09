"""Compare different agents against each other."""

import sys
from pathlib import Path
from typing import Dict, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import tyro

from src.agents import RandomAgent, HeuristicAgent, SmartHeuristicAgent
from src.utils.match import play_match


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
            
            wins1, draws, wins2 = play_match(
                agent1, 
                agent2, 
                num_games=num_games, 
                seed=seed + i * 10 + j,
                randomize_first_player=False  # Fixed order: agent1 always goes first
            )
            
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
    tyro.cli(compare_agents)

