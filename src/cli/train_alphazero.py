#!/usr/bin/env python3
"""CLI for training AlphaZero on Connect4."""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from src.envs.connect4 import Connect4Game, build_state_dict, CONNECT4_ROWS, CONNECT4_COLS
from src.features.observation_builder import BoardChannels
from src.models.alphazero import Connect4AlphaZeroNetwork
from src.agents.alphazero import AlphaZeroAgent, AlphaZeroSample
from src.training.alphazero import AlphaZeroTrainer, AlphaZeroConfig, AlphaZeroTrainerCallback
from src.search.mcts import MCTSConfig, OptimizedMCTS, make_batched_evaluator
from src.agents.connect4.heuristic_agent import HeuristicAgent
from src.agents.connect4.smart_heuristic_agent import SmartHeuristicAgent


def eval_vs_opponent(agent, game, opponent, num_games: int = 100, evaluator=None,
                     mcts_sims: int = 200, seed: int = 0) -> dict:
    """Evaluate agent vs any opponent (random, heuristic, etc). Returns win/draw/loss dict."""
    config = MCTSConfig(num_simulations=mcts_sims, c_puct=1.4)
    obs_builder = BoardChannels(board_shape=(CONNECT4_ROWS, CONNECT4_COLS))
    agent.eval()
    if evaluator is None:
        evaluator = make_batched_evaluator(agent, game, build_state_dict, obs_builder)

    wins, draws, losses = 0, 0, 0
    for game_idx in range(num_games):
        rng = np.random.default_rng(seed + game_idx)
        mcts = OptimizedMCTS(game, evaluator, config, rng=rng, batch_size=48)
        az_player_index = game_idx % 2
        az_token = [1, -1][az_player_index]
        state = game.initial_state()
        while not game.is_terminal(state):
            legal = list(game.legal_actions(state))
            legal_mask = np.zeros(CONNECT4_COLS, dtype=bool)
            for a in legal:
                legal_mask[a] = True
            if state.current_player_index == az_player_index:
                root = mcts.search(state, add_dirichlet_noise=False)
                action, _ = mcts.get_action_probs(root, temperature=0.0)
            else:
                state_dict = build_state_dict(state, game, legal_mask=legal_mask)
                obs = obs_builder.build(state_dict)
                action = opponent.act(obs, legal_mask=legal_mask, deterministic=False)
            state = game.apply_action(state, action)
        winner = game.winner(state)
        if winner == az_token:
            wins += 1
        elif winner == 0:
            draws += 1
        else:
            losses += 1
    return {"wins": wins, "draws": draws, "losses": losses}


def eval_vs_random(agent, game, num_games: int = 100, evaluator=None) -> float:
    """Convenience wrapper returning winrate vs random."""
    from src.agents.random_agent import RandomAgent
    rng = np.random.default_rng(0)

    class _Random:
        def act(self, obs, legal_mask=None, deterministic=False):
            legal = np.flatnonzero(legal_mask).tolist()
            return int(rng.choice(legal))

    result = eval_vs_opponent(agent, game, _Random(), num_games, evaluator)
    return result["wins"] / num_games


class PrintCallback(AlphaZeroTrainerCallback):

    def __init__(self, game, eval_every: int = 5, num_eval_games: int = 50, log_file: str = None):
        self.game = game
        self.eval_every = eval_every
        self.num_eval_games = num_eval_games
        self._evaluator = None
        self._heuristic = HeuristicAgent(seed=0)
        self._smart_heuristic = SmartHeuristicAgent(seed=0)
        self._log_f = None
        if log_file:
            import os
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            self._log_f = open(log_file, "w", buffering=1)
            self._log_f.write("iter,total_games,loss,policy_loss,value_loss,sp_time,train_time,wr_heuristic\n")

    def _get_evaluator(self, trainer):
        if self._evaluator is None:
            obs_builder = BoardChannels(board_shape=(CONNECT4_ROWS, CONNECT4_COLS))
            self._evaluator = make_batched_evaluator(
                trainer.agent, self.game, build_state_dict, obs_builder
            )
        return self._evaluator

    def on_iteration_start(self, trainer, iteration):
        print(f"\n--- Iteration {iteration} (games={trainer.total_games}) ---", flush=True)

    def on_self_play_end(self, trainer, iteration, games_played, samples_collected):
        print(f"  self-play: {games_played} games, {samples_collected} samples, buf={len(trainer.agent.replay_buffer)}", flush=True)

    def on_iteration_end(self, trainer, iteration, metrics):
        loss = metrics.get('avg_loss', 0)
        pl = metrics.get('avg_policy_loss', 0)
        vl = metrics.get('avg_value_loss', 0)
        sp_t = metrics.get('self_play_time', 0)
        tr_t = metrics.get('train_time', 0)
        print(f"  loss={loss:.4f} (p={pl:.4f} v={vl:.4f})  sp={sp_t:.1f}s train={tr_t:.1f}s", flush=True)

        wr_heuristic = None
        if self.eval_every > 0 and iteration % self.eval_every == 0:
            ev = self._get_evaluator(trainer)
            n = self.num_eval_games

            r_heur = eval_vs_opponent(trainer.agent, self.game, self._heuristic, n, ev, seed=iteration)
            wr_heuristic = r_heur["wins"] / n

            print(f"  >> vs heuristic: {wr_heuristic:.1%}  ({r_heur['wins']}W {r_heur['draws']}D {r_heur['losses']}L)", flush=True)

        if self._log_f:
            def fmt(v): return f"{v:.4f}" if v is not None else ""
            self._log_f.write(
                f"{iteration},{trainer.total_games},{loss:.6f},{pl:.6f},{vl:.6f},"
                f"{sp_t:.2f},{tr_t:.2f},{fmt(wr_heuristic)}\n"
            )


def horizontal_flip_augment(sample: AlphaZeroSample):
    """Augment sample by horizontal flip."""
    flipped_obs = np.flip(sample.observation, axis=-1).copy()
    flipped_mask = np.flip(sample.legal_mask).copy()
    flipped_policy = np.flip(sample.target_policy).copy()

    return [
        AlphaZeroSample(
            observation=flipped_obs,
            legal_mask=flipped_mask,
            target_policy=flipped_policy,
            target_value=sample.target_value,
        )
    ]


def main():
    parser = argparse.ArgumentParser(description="Train AlphaZero on Connect4")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    parser.add_argument("--games-per-iter", type=int, default=25, help="Games per iteration")
    parser.add_argument("--mcts-sims", type=int, default=100, help="MCTS simulations per move")
    parser.add_argument("--mcts-batch", type=int, default=48, help="MCTS leaf batch size")
    parser.add_argument("--train-steps", type=int, default=100, help="Training steps per iteration")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate")
    parser.add_argument("--trunk-channels", type=int, default=64, help="Network trunk channels")
    parser.add_argument("--res-blocks", type=int, default=4, help="Number of residual blocks")
    parser.add_argument("--checkpoint-dir", type=str, default="runs/alphazero_connect4", help="Checkpoint directory")
    parser.add_argument("--checkpoint-interval", type=int, default=5, help="Save checkpoint every N iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-augment", action="store_true", help="Disable augmentation")
    parser.add_argument("--cuda-graph", action="store_true", help="Use CUDA Graphs for ~1.5x faster inference")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--eval-every", type=int, default=5, help="Evaluate vs random every N iterations (0=off)")
    parser.add_argument("--log-file", type=str, default=None, help="CSV log file path (e.g. runs/run1/log.csv)")

    args = parser.parse_args()

    np.random.seed(args.seed)

    game = Connect4Game()

    observation_builder = BoardChannels(
        board_shape=(CONNECT4_ROWS, CONNECT4_COLS),
        include_last_move=False,
        include_legal_moves=False,
    )

    network = Connect4AlphaZeroNetwork(
        rows=CONNECT4_ROWS,
        cols=CONNECT4_COLS,
        in_channels=2,
        num_actions=CONNECT4_COLS,
        trunk_channels=args.trunk_channels,
        num_res_blocks=args.res_blocks,
    )

    agent = AlphaZeroAgent(
        network=network,
        num_actions=CONNECT4_COLS,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )

    config = AlphaZeroConfig(
        num_games_per_iteration=args.games_per_iter,
        mcts_simulations=args.mcts_sims,
        mcts_batch_size=args.mcts_batch,
        train_steps_per_iteration=args.train_steps,
        batch_size=args.batch_size,
        num_iterations=args.iterations,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        use_cuda_graph=args.cuda_graph,
    )

    augment_fn = None if args.no_augment else horizontal_flip_augment

    log_file = args.log_file or f"{args.checkpoint_dir}/log.csv"

    trainer = AlphaZeroTrainer(
        game=game,
        agent=agent,
        observation_builder=observation_builder,
        state_to_dict_fn=build_state_dict,
        initial_state_fn=game.initial_state,
        config=config,
        callbacks=[PrintCallback(game, eval_every=args.eval_every, log_file=log_file)],
        augment_fn=augment_fn,
        rng=np.random.default_rng(args.seed),
    )

    print(f"AlphaZero Connect4 | {args.iterations} iters | {args.games_per_iter} games/iter | {args.mcts_sims} MCTS sims", flush=True)
    print(f"Network: {args.trunk_channels}ch x {args.res_blocks} res blocks | device={agent.device}", flush=True)
    print(f"Augmentation: {'off' if args.no_augment else 'on'} | CUDA Graphs: {'on' if args.cuda_graph else 'off'}", flush=True)
    print(flush=True)

    trainer.train()

    print(flush=True)
    final = eval_vs_random(agent, game, 200)
    print(f"Final winrate vs random: {final:.1%}", flush=True)


if __name__ == "__main__":
    main()
