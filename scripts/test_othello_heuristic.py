"""Test Othello heuristic agent vs random agent."""

import sys
sys.path.insert(0, "/home/holidin/projects/RL")

from src.envs.othello import OthelloEnv
from src.agents.random_agent import RandomAgent
from src.agents.othello import OthelloHeuristicAgent


def play_game(agent1, agent2, render=False):
    """Play a single game and return the winner."""
    env = OthelloEnv()
    obs = env.reset()
    
    if render:
        env.render()
    
    while not env.done:
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            break
        
        current_player = env.current_player()
        agent = agent1 if current_player == 0 else agent2
        
        action = agent.act(obs, legal_mask=env.legal_actions_mask)
        result = env.step(action)
        
        if render:
            print(f"\nPlayer {current_player} plays action {action}")
            env.render()
        
        obs = result.obs
        
        if result.terminated:
            return result.info['winner']
    
    return 0


def main():
    """Run multiple games and report statistics."""
    heuristic = OthelloHeuristicAgent()
    random = RandomAgent()
    
    num_games = 20
    heuristic_wins = 0
    random_wins = 0
    draws = 0
    
    print(f"Playing {num_games} games: Heuristic vs Random")
    print("=" * 50)
    
    for i in range(num_games):
        if i % 2 == 0:
            winner = play_game(heuristic, random, render=(i == 0))
            if winner == 1:
                heuristic_wins += 1
            elif winner == -1:
                random_wins += 1
            else:
                draws += 1
        else:
            winner = play_game(random, heuristic, render=False)
            if winner == -1:
                heuristic_wins += 1
            elif winner == 1:
                random_wins += 1
            else:
                draws += 1
        
        if (i + 1) % 5 == 0:
            print(f"After {i + 1} games: Heuristic {heuristic_wins}, Random {random_wins}, Draws {draws}")
    
    print("\n" + "=" * 50)
    print(f"Final results after {num_games} games:")
    print(f"  Heuristic wins: {heuristic_wins} ({100 * heuristic_wins / num_games:.1f}%)")
    print(f"  Random wins: {random_wins} ({100 * random_wins / num_games:.1f}%)")
    print(f"  Draws: {draws} ({100 * draws / num_games:.1f}%)")


if __name__ == "__main__":
    main()
