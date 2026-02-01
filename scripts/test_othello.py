"""Simple script to test Othello environment."""

import sys
sys.path.insert(0, "/home/holidin/projects/RL")

from src.envs.othello import OthelloEnv
from src.agents.random_agent import RandomAgent


def test_random_game():
    """Play a random game to completion."""
    env = OthelloEnv()
    agent1 = RandomAgent()
    agent2 = RandomAgent()
    
    obs = env.reset()
    env.render()
    
    move_count = 0
    while not env.done:
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            break
        
        current_player = env.current_player()
        agent = agent1 if current_player == 0 else agent2
        
        action = agent.act(obs, legal_mask=env.legal_actions_mask)
        result = env.step(action)
        
        move_count += 1
        print(f"\nMove {move_count}: Player {current_player} plays action {action}")
        env.render()
        
        obs = result.obs
        
        if result.terminated:
            print(f"\nGame over! Winner: {result.info['winner']}")
            print(f"Termination reason: {result.info['termination_reason']}")
            break
    
    print(f"\nTotal moves: {move_count}")


if __name__ == "__main__":
    test_random_game()
