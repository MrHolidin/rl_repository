#!/usr/bin/env python3
"""Train AlphaZero on TicTacToe and evaluate vs random."""

import argparse

import numpy as np

from src.envs.tictactoe import TicTacToeGame, build_state_dict
from src.features.observation_builder import BoardChannels
from src.models.alphazero import TicTacToeAlphaZeroNetwork
from src.agents.alphazero import AlphaZeroAgent
from src.training.alphazero import AlphaZeroTrainer, AlphaZeroConfig, AlphaZeroTrainerCallback
from src.search.mcts import MCTSConfig, OptimizedMCTS, make_batched_evaluator


def eval_vs_random(agent: AlphaZeroAgent, game: TicTacToeGame, num_games: int = 100,
                   evaluator=None) -> float:
    """Play agent (MCTS) vs random. Returns win rate."""
    config = MCTSConfig(num_simulations=50, c_puct=1.4)

    agent.eval()
    if evaluator is None:
        obs_builder = BoardChannels(board_shape=(3, 3))
        evaluator = make_batched_evaluator(agent, game, build_state_dict, obs_builder)

    wins, draws, losses = 0, 0, 0

    for seed in range(num_games):
        rng = np.random.default_rng(seed)
        mcts = OptimizedMCTS(game, evaluator, config, rng=rng, batch_size=9)
        state = game.initial_state()

        # Alternate who goes first
        mcts_player_index = seed % 2

        while not game.is_terminal(state):
            if state.current_player_index == mcts_player_index:
                root = mcts.search(state, add_dirichlet_noise=False)
                action, _ = mcts.get_action_probs(root, temperature=0.0)
            else:
                action = rng.choice(list(game.legal_actions(state)))
            state = game.apply_action(state, action)

        winner = game.winner(state)
        mcts_token = [1, -1][mcts_player_index]
        if winner == mcts_token:
            wins += 1
        elif winner == 0:
            draws += 1
        else:
            losses += 1

    return wins / num_games


class EvalCallback(AlphaZeroTrainerCallback):

    def __init__(self, game: TicTacToeGame, eval_every: int = 5, num_eval_games: int = 100):
        self.game = game
        self.eval_every = eval_every
        self.num_eval_games = num_eval_games
        self._evaluator = None

    def on_iteration_start(self, trainer, iteration):
        print(f"\n--- Iteration {iteration} ---", flush=True)

    def on_self_play_end(self, trainer, iteration, games_played, samples_collected):
        print(f"  self-play: {games_played} games, {samples_collected} samples, buf={len(trainer.agent.replay_buffer)}", flush=True)

    def on_iteration_end(self, trainer, iteration, metrics):
        loss = metrics.get("avg_loss", 0)
        pl = metrics.get("avg_policy_loss", 0)
        vl = metrics.get("avg_value_loss", 0)
        sp_t = metrics.get("self_play_time", 0)
        tr_t = metrics.get("train_time", 0)
        print(f"  loss={loss:.4f} (p={pl:.4f} v={vl:.4f})  sp={sp_t:.1f}s train={tr_t:.1f}s", flush=True)

        if iteration % self.eval_every == 0:
            # Reuse evaluator across eval calls (avoids recreating it)
            if self._evaluator is None:
                obs_builder = BoardChannels(board_shape=(3, 3))
                self._evaluator = make_batched_evaluator(
                    trainer.agent, self.game, build_state_dict, obs_builder
                )
            winrate = eval_vs_random(trainer.agent, self.game, self.num_eval_games, self._evaluator)
            print(f"  >> winrate vs random: {winrate:.1%}  (iter {iteration})", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train AlphaZero on TicTacToe")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--games-per-iter", type=int, default=50)
    parser.add_argument("--mcts-sims", type=int, default=100)
    parser.add_argument("--mcts-batch", type=int, default=9)
    parser.add_argument("--max-train-steps", type=int, default=100)
    parser.add_argument("--max-kl", type=float, default=None, help="KL early stop threshold")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--trunk-channels", type=int, default=64)
    parser.add_argument("--res-blocks", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)

    game = TicTacToeGame()

    obs_builder = BoardChannels(board_shape=(3, 3))

    network = TicTacToeAlphaZeroNetwork(
        trunk_channels=args.trunk_channels,
        num_res_blocks=args.res_blocks,
    )

    agent = AlphaZeroAgent(
        network=network,
        num_actions=9,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )

    config = AlphaZeroConfig(
        num_games_per_iteration=args.games_per_iter,
        mcts_simulations=args.mcts_sims,
        mcts_batch_size=args.mcts_batch,
        max_train_steps_per_iteration=args.max_train_steps,
        max_kl_divergence=args.max_kl,
        num_iterations=args.iterations,
        checkpoint_interval=0,  # no checkpoints
        dirichlet_alpha=1.0,  # 10/num_actions for TicTacToe
        dirichlet_frac=0.25,
        temperature_threshold=5,
    )

    trainer = AlphaZeroTrainer(
        game=game,
        agent=agent,
        observation_builder=obs_builder,
        state_to_dict_fn=build_state_dict,
        initial_state_fn=game.initial_state,
        config=config,
        callbacks=[EvalCallback(game, eval_every=args.eval_every)],
        rng=np.random.default_rng(args.seed),
    )

    print(f"AlphaZero TicTacToe | {args.iterations} iters | {args.games_per_iter} games/iter | {args.mcts_sims} MCTS sims", flush=True)
    print(f"Network: {args.trunk_channels}ch x {args.res_blocks} res blocks | device={agent.device}", flush=True)
    print(flush=True)

    trainer.train()

    print(flush=True)
    final = eval_vs_random(agent, game, 200)
    print(f"Final winrate vs random: {final:.1%}", flush=True)


if __name__ == "__main__":
    main()
