"""
Evaluate AlphaZero checkpoint on Pons test-set positions.

Test sets from https://blog.gamesolver.org/data/:
  Test_L3_R1  – End-Easy   (>28 moves played, <14 remaining)
  Test_L1_R1  – Begin-Easy (<=14 moves played, <14 remaining)

Position format: sequence of 1-indexed columns, then space, then oracle score.
Oracle score > 0 means current player wins, 0 = draw, < 0 means current player loses.

Move quality check:
  After the model picks a move, we query the Pons solver for the resulting position.
  A move is "correct" if it doesn't worsen the outcome category (win/draw/loss).
    - From a winning position: move is correct if resulting score is still a win
      (i.e. opponent faces score < 0, equivalently negated score > 0, but
       Pons score of resulting position from opponent POV ≤ 0 → current stayed winning).
    - From a drawing position: move is correct if resulting score is still 0.
    - From a losing position: any move; we track if the model picks the least-bad one.
"""

import argparse
import random
import subprocess
import sys
import time

import numpy as np

from src.agents.alphazero.agent import AlphaZeroAgent
from src.envs.connect4 import Connect4Game, build_state_dict, CONNECT4_ROWS, CONNECT4_COLS
from src.features.observation_builder import BoardChannels
from src.search.mcts import MCTSConfig, OptimizedMCTS, make_batched_evaluator

SOLVER = "/tmp/pons_c4/c4solver"


def oracle_score(seq: str) -> int:
    """Query the Pons solver for the score of a position given as move sequence."""
    result = subprocess.run(
        [SOLVER], input=seq + "\n", capture_output=True, text=True
    )
    # output: "<seq> <score>\n"  (plus possible warning on stderr)
    out = result.stdout.strip().split()
    return int(out[-1])


def replay_position(seq: str, game: Connect4Game):
    """Replay a Pons move sequence (1-indexed columns) → Connect4State."""
    state = game.initial_state()
    for ch in seq:
        col = int(ch) - 1  # 0-indexed
        state = game.apply_action(state, col)
    return state


def load_test_set(path: str):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            entries.append((parts[0], int(parts[1])))
    return entries


def sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def evaluate(
    checkpoint: str,
    test_files: list,
    mcts_sims: int,
    device: str,
    seed: int,
):
    print(f"Loading {checkpoint} ...", flush=True)
    agent = AlphaZeroAgent.load(checkpoint, device=device)
    agent.eval()

    game = Connect4Game()
    obs_builder = BoardChannels(board_shape=(CONNECT4_ROWS, CONNECT4_COLS))
    evaluator = make_batched_evaluator(agent, game, build_state_dict, obs_builder)

    config = MCTSConfig(num_simulations=mcts_sims, c_puct=1.4)
    mcts = OptimizedMCTS(game, evaluator, config, batch_size=32)

    # warmup
    mcts.search(game.initial_state(), add_dirichlet_noise=False)

    rng = random.Random(seed)
    all_move_accs = []

    for path, set_name, n_per_set in test_files:
        entries = load_test_set(path)
        sample = rng.sample(entries, min(n_per_set, len(entries)))

        # move quality: did the model preserve the outcome after its move?
        # rows = oracle category; cols = resulting category after model's move
        move_confusion = {1: {1: 0, 0: 0, -1: 0},
                          0: {1: 0, 0: 0, -1: 0},
                         -1: {1: 0, 0: 0, -1: 0}}

        t0 = time.time()
        for seq, oracle_before in sample:
            state = replay_position(seq, game)
            root = mcts.search(state, add_dirichlet_noise=False)

            # best move (greedy, 0-indexed) → convert to 1-indexed for Pons
            action, _ = mcts.get_action_probs(root, temperature=0.0)
            new_seq = seq + str(action + 1)

            # oracle score of resulting position (from opponent's POV → negate for current player)
            score_after_opp = oracle_score(new_seq)
            # after move, it's opponent's turn: their score is score_after_opp
            # from current player's perspective the outcome is -score_after_opp
            outcome_for_current = -score_after_opp

            move_confusion[sign(oracle_before)][sign(outcome_for_current)] += 1

        elapsed = time.time() - t0
        n = len(sample)

        correct_moves = sum(move_confusion[s][s] for s in (1, 0, -1))
        # for losing positions, any move is "not worse than losing" — exclude from accuracy
        n_non_losing = sum(sum(move_confusion[s].values()) for s in (1, 0))
        move_acc = correct_moves / n if n else 0.0

        def row(label, osign):
            d = move_confusion[osign]
            total = sum(d.values())
            if total == 0:
                return
            ok = d[osign]
            blunder_win  = d[1]  if osign != 1  else 0
            blunder_draw = d[0]  if osign != 0  else 0
            blunder_loss = d[-1] if osign != -1 else 0
            print(f"    from {label:5s} (n={total:2d}):  "
                  f"preserved={ok:2d}  "
                  f"→win={blunder_win}  →draw={blunder_draw}  →loss={blunder_loss}")

        print(f"\n[{set_name}]  n={n}  mcts_sims={mcts_sims}  elapsed={elapsed:.1f}s")
        print(f"  move_outcome_preserved = {move_acc:.1%}  ({correct_moves}/{n})")
        row("WIN",  1)
        row("DRAW", 0)
        row("LOSS", -1)

        all_move_accs.append(move_acc)

    overall = sum(all_move_accs) / len(all_move_accs)
    print(f"\nOverall move_outcome_preserved: {overall:.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?",
                        default="runs/alphazero/connect4/long_run/alphazero_iter_000100.pt")
    parser.add_argument("--end-easy",      default="/tmp/Test_L3_R1.txt")
    parser.add_argument("--middle-easy",   default="/tmp/Test_L2_R1.txt")
    parser.add_argument("--middle-medium", default="/tmp/Test_L2_R2.txt")
    parser.add_argument("--begin-easy",    default="/tmp/Test_L1_R1.txt")
    parser.add_argument("--begin-medium",  default="/tmp/Test_L1_R2.txt")
    parser.add_argument("--begin-hard",    default="/tmp/Test_L1_R3.txt")
    parser.add_argument("--n",          type=int, default=50,
                        help="positions per test set")
    parser.add_argument("--n-hard",     type=int, default=20,
                        help="positions for Begin-Hard (solver is slow there)")
    parser.add_argument("--mcts-sims",  type=int, default=200)
    parser.add_argument("--device",     default=None)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    test_files = [
        (args.end_easy,      "End-Easy      (>28 moves, <14 rem)",   args.n),
        (args.middle_easy,   "Middle-Easy   (14-28 moves, <14 rem)", args.n),
        (args.middle_medium, "Middle-Medium (14-28 moves, 14-28 rem)", args.n),
        (args.begin_easy,    "Begin-Easy    (<=14 moves, <14 rem)",  args.n),
        (args.begin_medium,  "Begin-Medium  (<=14 moves, 14-28 rem)", args.n),
        (args.begin_hard,    "Begin-Hard    (<=14 moves, >=28 rem)", args.n_hard),
    ]

    evaluate(
        checkpoint=args.checkpoint,
        test_files=test_files,
        mcts_sims=args.mcts_sims,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
