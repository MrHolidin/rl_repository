#!/usr/bin/env python3
"""CLI for training AlphaZero on Connect4."""

import argparse
from pathlib import Path

import numpy as np

from src.envs.connect4 import Connect4Game, build_state_dict, CONNECT4_ROWS, CONNECT4_COLS
from src.features.observation_builder import BoardChannels
from src.models.alphazero import Connect4AlphaZeroNetwork
from src.agents.alphazero import AlphaZeroAgent
from src.training.alphazero import AlphaZeroTrainer, AlphaZeroConfig, AlphaZeroTrainerCallback
from src.search.mcts import MCTSConfig, OptimizedMCTS, make_batched_evaluator
from src.agents.connect4.heuristic_agent import HeuristicAgent
from src.training.alphazero_augmentations import horizontal_flip_augment


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


_CSV_HEADER = (
    "iter,total_games,loss,policy_loss,value_loss,sp_time,train_time,"
    "fresh_frac,avg_age_iters,policy_kl,train_steps,kl_stopped,wr_heuristic,"
    "search_kl,top1_agreement,mass_on_best,target_entropy_norm,"
    "val_policy_loss,val_value_loss,"
    "probe_p_start,probe_p_end,probe_v_start,probe_v_end,"
    "policy_research_kl,policy_research_top1,value_research_absdiff,re_search_n,"
    "opponent_pool_games\n"
)


def _compact_trace(values: list, max_shown: int = 10) -> str:
    """Return a compact string representation of a loss trace."""
    if not values:
        return ""
    if len(values) <= max_shown:
        return " ".join(f"{v:.3f}" for v in values)
    step = len(values) / max_shown
    indices = sorted({int(i * step) for i in range(max_shown - 1)} | {len(values) - 1})
    return " ".join(f"{values[i]:.3f}" for i in indices) + f" ({len(values)} steps)"


class PrintCallback(AlphaZeroTrainerCallback):

    def __init__(self, game, eval_every: int = 5, num_eval_games: int = 50, log_file: str = None):
        self.game = game
        self.eval_every = eval_every
        self.num_eval_games = num_eval_games
        self._evaluator = None
        self._heuristic = HeuristicAgent(seed=0)
        self._log_f = None
        if log_file:
            import os
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            self._log_f = open(log_file, "w", buffering=1)
            self._log_f.write(_CSV_HEADER)

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
        loss        = metrics.get("avg_loss", 0)
        pl          = metrics.get("avg_policy_loss", 0)
        vl          = metrics.get("avg_value_loss", 0)
        sp_t        = metrics.get("self_play_time", 0)
        tr_t        = metrics.get("train_time", 0)
        fresh       = metrics.get("fresh_fraction", 0)
        age         = metrics.get("avg_age_iters", 0)
        kl          = metrics.get("policy_kl", None)
        train_steps = metrics.get("train_steps", 0)
        kl_stopped  = metrics.get("kl_stopped", False)
        kl_trace    = metrics.get("kl_trace", [])

        # Search quality
        s_kl        = metrics.get("search_kl", None)
        top1        = metrics.get("top1_agreement", None)
        mass_best   = metrics.get("mass_on_best", None)
        ent_norm    = metrics.get("target_entropy_norm", None)

        # Val losses
        val_pl      = metrics.get("val_policy_loss", None)
        val_vl      = metrics.get("val_value_loss", None)

        # Per-step traces
        p_trace     = metrics.get("policy_loss_trace", [])
        v_trace     = metrics.get("value_loss_trace", [])

        # Re-search stability (optional)
        pr_kl       = metrics.get("policy_research_kl", None)
        pr_top1     = metrics.get("policy_research_top1", None)
        v_rd        = metrics.get("value_research_absdiff", None)
        re_n        = metrics.get("re_search_n", None)
        opp_pool_n  = metrics.get("opponent_pool_games", 0)

        kl_str   = f"  kl={kl:.4f}" if kl is not None else ""
        stop_str = " [kl_stop]" if kl_stopped else ""
        print(
            f"  loss={loss:.4f} (p={pl:.4f} v={vl:.4f})  sp={sp_t:.1f}s train={tr_t:.1f}s  "
            f"fresh={fresh:.1%} age={age:.1f}iters  steps={train_steps}{stop_str}{kl_str}",
            flush=True,
        )

        if kl_trace:
            trace_str = " → ".join(f"{v:.3f}" for v in kl_trace)
            if kl_stopped:
                trace_str += " [STOP]"
            print(f"  kl_trace: {trace_str}", flush=True)

        if s_kl is not None:
            print(
                f"  search: kl={s_kl:.4f}  top1={top1:.3f}  mass_best={mass_best:.3f}  "
                f"ent_norm={ent_norm:.3f}",
                flush=True,
            )

        if val_pl is not None:
            print(f"  val:    p_loss={val_pl:.4f}  v_loss={val_vl:.4f}", flush=True)

        if p_trace:
            print(f"  probe_p: {_compact_trace(p_trace)}", flush=True)
        if v_trace:
            print(f"  probe_v: {_compact_trace(v_trace)}", flush=True)

        if pr_kl is not None:
            print(
                f"  re-search: sym_kl={pr_kl:.4f}  top1={pr_top1:.3f}  |dv|={v_rd:.4f}  (n={re_n})",
                flush=True,
            )

        if opp_pool_n:
            print(f"  opponent_pool_games: {opp_pool_n}/{metrics.get('games_played', 0)}", flush=True)

        wr_heuristic = None
        if self.eval_every > 0 and iteration % self.eval_every == 0:
            ev = self._get_evaluator(trainer)
            n = self.num_eval_games

            r_heur = eval_vs_opponent(trainer.agent, self.game, self._heuristic, n, ev, seed=iteration)
            wr_heuristic = r_heur["wins"] / n

            print(f"  >> vs heuristic: {wr_heuristic:.1%}  ({r_heur['wins']}W {r_heur['draws']}D {r_heur['losses']}L)", flush=True)

        if self._log_f:
            def fmt(v): return f"{v:.6f}" if v is not None else ""
            probe_p_start = fmt(p_trace[0]) if p_trace else ""
            probe_p_end   = fmt(p_trace[-1]) if p_trace else ""
            probe_v_start = fmt(v_trace[0]) if v_trace else ""
            probe_v_end   = fmt(v_trace[-1]) if v_trace else ""
            rsn = str(int(re_n)) if re_n is not None else ""
            self._log_f.write(
                f"{iteration},{trainer.total_games},{loss:.6f},{pl:.6f},{vl:.6f},"
                f"{sp_t:.2f},{tr_t:.2f},{fresh:.4f},{age:.2f},{fmt(kl)},{train_steps},{int(kl_stopped)},{fmt(wr_heuristic)},"
                f"{fmt(s_kl)},{fmt(top1)},{fmt(mass_best)},{fmt(ent_norm)},"
                f"{fmt(val_pl)},{fmt(val_vl)},"
                f"{probe_p_start},{probe_p_end},{probe_v_start},{probe_v_end},"
                f"{fmt(pr_kl)},{fmt(pr_top1)},{fmt(v_rd)},{rsn},{opp_pool_n}\n"
            )



def main():
    parser = argparse.ArgumentParser(description="Train AlphaZero on Connect4")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    parser.add_argument("--games-per-iter", type=int, default=25, help="Games per iteration")
    parser.add_argument("--mcts-sims", type=int, default=100, help="MCTS simulations per move")
    parser.add_argument("--mcts-batch", type=int, default=256, help="MCTS leaf batch size")
    parser.add_argument(
        "--temperature-threshold",
        type=int,
        default=15,
        help="Self-play: use temperature=1 on visit policy for the first N plies, then 0 (greedy). E.g. 8.",
    )
    parser.add_argument("--max-train-steps", type=int, default=100, help="Max training steps per iteration")
    parser.add_argument("--max-kl", type=float, default=None, help="Stop training steps early if policy KL exceeds this (e.g. 0.4)")
    parser.add_argument("--kl-warmup", type=int, default=3, help="Number of initial iterations without KL early stop")
    parser.add_argument("--replay-buffer-size", type=int, default=100_000, help="Replay buffer capacity")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate")
    parser.add_argument("--trunk-channels", type=int, default=64, help="Network trunk channels")
    parser.add_argument("--res-blocks", type=int, default=4, help="Number of residual blocks")
    parser.add_argument(
        "--run-dir",
        type=str,
        default="runs/alphazero_connect4",
        help="Working directory: checkpoints, opponent-pool .pt scan, default log at <run-dir>/log.csv",
    )
    parser.add_argument("--checkpoint-dir", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        metavar="N",
        help="Save alphazero_iter_*.pt every N training iterations (0 = never; opponent pool updates on each save).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-augment", action="store_true", help="Disable augmentation")
    parser.add_argument("--cuda-graph", action="store_true", help="Use CUDA Graphs for ~1.5x faster inference")
    parser.add_argument("--game-pool", type=int, default=1, help="Number of concurrent self-play games (game pool size, default=1)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--eval-every", type=int, default=5, help="Evaluate vs random every N iterations (0=off)")
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="CSV log path (default: <run-dir>/log.csv if omitted)",
    )
    parser.add_argument("--val-split", type=float, default=0.0, help="Fraction of self-play samples held out for val loss (0=off, e.g. 0.1)")
    parser.add_argument(
        "--re-search-stability",
        type=int,
        default=0,
        help="Re-search stability: sample this many roots from self-play, run MCTS twice each (0=off). Try 4–8.",
    )
    parser.add_argument(
        "--re-search-stability-sims",
        type=int,
        default=None,
        help="Sims per diagnostic MCTS (default: same as --mcts-sims). Lower for cheaper logging.",
    )
    parser.add_argument(
        "--opponent-pool-frac",
        type=float,
        default=0.0,
        help="Fraction of games vs MCTS opponents from last X own checkpoints in run-dir (requires --game-pool 1).",
    )
    parser.add_argument(
        "--opponent-pool-size",
        type=int,
        default=5,
        help="Most recent alphazero_iter_*.pt files in run-dir used as opponents (default 5).",
    )

    args = parser.parse_args()

    if args.checkpoint_dir is not None:
        if args.run_dir != "runs/alphazero_connect4" and Path(
            args.run_dir
        ) != Path(args.checkpoint_dir):
            parser.error("--run-dir and --checkpoint-dir disagree; pass one path only.")
        run_dir = args.checkpoint_dir
    else:
        run_dir = args.run_dir

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
        replay_buffer_size=args.replay_buffer_size,
        device=args.device,
        seed=args.seed,
    )

    config = AlphaZeroConfig(
        num_games_per_iteration=args.games_per_iter,
        mcts_simulations=args.mcts_sims,
        mcts_batch_size=args.mcts_batch,
        temperature_threshold=args.temperature_threshold,
        max_train_steps_per_iteration=args.max_train_steps,
        max_kl_divergence=args.max_kl,
        kl_warmup_iterations=args.kl_warmup,
        num_iterations=args.iterations,
        checkpoint_dir=run_dir,
        checkpoint_interval=args.checkpoint_interval,
        use_cuda_graph=args.cuda_graph,
        game_pool_size=args.game_pool,
        val_split_fraction=args.val_split,
        re_search_stability_positions=args.re_search_stability,
        re_search_stability_num_sims=args.re_search_stability_sims,
        opponent_pool_frac=args.opponent_pool_frac,
        opponent_pool_size=args.opponent_pool_size,
    )

    augment_fn = None if args.no_augment else horizontal_flip_augment

    log_file = args.log_file or str(Path(run_dir) / "log.csv")

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

    ckpt_msg = (
        f"checkpoint every {args.checkpoint_interval} iter(s)"
        if args.checkpoint_interval > 0
        else "checkpoints off"
    )
    print(
        f"AlphaZero Connect4 | run_dir={run_dir} | {args.iterations} iters | "
        f"{args.games_per_iter} games/iter | {args.mcts_sims} MCTS sims | {ckpt_msg}",
        flush=True,
    )
    print(f"Log: {log_file}", flush=True)
    print(f"Network: {args.trunk_channels}ch x {args.res_blocks} res blocks | device={agent.device}", flush=True)
    print(
        f"Augmentation: {'off' if args.no_augment else 'on'} | CUDA Graphs: {'on' if args.cuda_graph else 'off'} "
        f"| game_pool={args.game_pool} | temp_thresh_plies={args.temperature_threshold}"
        + (
            f" | opponent_pool: last {args.opponent_pool_size} ckpt, frac={args.opponent_pool_frac}"
            if args.opponent_pool_frac > 0
            else ""
        ),
        flush=True,
    )
    print(flush=True)

    trainer.train()

    print(flush=True)
    final = eval_vs_random(agent, game, 200)
    print(f"Final winrate vs random: {final:.1%}", flush=True)


if __name__ == "__main__":
    main()
