#!/usr/bin/env python3
"""Evaluate trained AlphaZero checkpoint vs heuristic opponents on Connect4."""

import argparse
from typing import Optional

import numpy as np

from src.envs.connect4 import Connect4Game, build_state_dict, CONNECT4_ROWS, CONNECT4_COLS
from src.features.observation_builder import BoardChannels
from src.agents.alphazero.agent import AlphaZeroAgent
from src.agents.connect4.heuristic_agent import HeuristicAgent
from src.agents.connect4.smart_heuristic_agent import SmartHeuristicAgent
from src.search.mcts import MCTSConfig, OptimizedMCTS, make_batched_evaluator


def play_match(
    az_agent: AlphaZeroAgent,
    evaluator,
    opponent,
    game: Connect4Game,
    obs_builder: BoardChannels,
    num_games: int,
    mcts_sims: int,
    temperature: float,
    seed: int = 0,
) -> dict:
    """
    Play num_games between AlphaZero and opponent, alternating who goes first.
    Returns win/draw/loss counts from AlphaZero's perspective.
    """
    config = MCTSConfig(num_simulations=mcts_sims, c_puct=1.4)
    az_agent.eval()

    wins, draws, losses = 0, 0, 0

    for game_idx in range(num_games):
        rng = np.random.default_rng(seed + game_idx)
        mcts = OptimizedMCTS(game, evaluator, config, rng=rng, batch_size=48)

        # Alternate who plays first
        az_player_index = game_idx % 2  # 0 = X (+1), 1 = O (-1)
        az_token = [1, -1][az_player_index]

        state = game.initial_state()

        while not game.is_terminal(state):
            # Build canonical observation for both agents
            legal = list(game.legal_actions(state))
            legal_mask = np.zeros(CONNECT4_COLS, dtype=bool)
            for a in legal:
                legal_mask[a] = True

            state_dict = build_state_dict(state, game, legal_mask=legal_mask)
            obs = obs_builder.build(state_dict)

            if state.current_player_index == az_player_index:
                root = mcts.search(state, add_dirichlet_noise=False)
                action, _ = mcts.get_action_probs(root, temperature=temperature)
            else:
                action = opponent.act(obs, legal_mask=legal_mask, deterministic=False)

            state = game.apply_action(state, action)

        winner = game.winner(state)
        if winner == az_token:
            wins += 1
        elif winner == 0:
            draws += 1
        else:
            losses += 1

    return {"wins": wins, "draws": draws, "losses": losses, "total": num_games}


def print_result(name: str, result: dict) -> None:
    n = result["total"]
    w, d, l = result["wins"], result["draws"], result["losses"]
    wr = w / n
    print(f"  vs {name:<20} {w:3d}W {d:3d}D {l:3d}L  winrate={wr:.1%}  ({n} games)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate AlphaZero vs heuristic opponents")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint .pt file")
    parser.add_argument("--num-games", type=int, default=100)
    parser.add_argument("--mcts-sims", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="MCTS temperature for non-determinism (0=greedy, 0.3=slight noise)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}", flush=True)
    agent = AlphaZeroAgent.load(args.checkpoint, device=args.device)
    agent.eval()

    game = Connect4Game()
    obs_builder = BoardChannels(board_shape=(CONNECT4_ROWS, CONNECT4_COLS))
    evaluator = make_batched_evaluator(agent, game, build_state_dict, obs_builder)

    print(f"AlphaZero eval | {args.num_games} games each | {args.mcts_sims} MCTS sims | temp={args.temperature}", flush=True)
    print(flush=True)

    # Warmup
    state = game.initial_state()
    config = MCTSConfig(num_simulations=10, c_puct=1.4)
    mcts_w = OptimizedMCTS(game, evaluator, config, batch_size=48)
    mcts_w.search(state, add_dirichlet_noise=False)
    print("GPU warmed up.", flush=True)
    print(flush=True)

    opponents = [
        ("Heuristic",       HeuristicAgent(seed=args.seed)),
        ("SmartHeuristic",  SmartHeuristicAgent(seed=args.seed)),
    ]

    for name, opp in opponents:
        print(f"Playing vs {name}...", flush=True)
        result = play_match(
            agent, evaluator, opp, game, obs_builder,
            num_games=args.num_games,
            mcts_sims=args.mcts_sims,
            temperature=args.temperature,
            seed=args.seed,
        )
        print_result(name, result)
        print(flush=True)


if __name__ == "__main__":
    main()
