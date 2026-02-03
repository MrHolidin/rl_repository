"""Find positions where depth-3 minimax fails.

Modes:
- Win-in-2: positions where we can force a win in 2 moves; depth-3 should find it (3 plies).
- Lose-in-2-avoid: positions where some moves lose in 2 (opponent forces win); depth-3 cannot see that (needs 4 plies).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.envs.connect4 import Connect4Game, Connect4State
from src.search.connect4 import make_connect4_heuristic_minimax_policy


def win_in_one_move(game: Connect4Game, state: Connect4State) -> set[int] | None:
    """Current player can win in 1 move. Return set of winning moves or None."""
    if game.is_terminal(state):
        return None
    us = game.current_player(state)
    legal = list(game.legal_actions(state))
    wins: set[int] = set()
    for a in legal:
        s1 = game.apply_action(state, a)
        if game.is_terminal(s1) and game.winner(s1) == us:
            wins.add(a)
    return wins if wins else None


def _opp_force_win_in_two_from_our_turn(game: Connect4Game, state: Connect4State, opp: int) -> bool:
    """From state (our turn), can opponent force win in 2 moves (opp, us, opp)?"""
    for a2 in game.legal_actions(state):
        s3 = game.apply_action(state, a2)
        if game.is_terminal(s3):
            continue
        opp_can_win_from_s3 = False
        for a_opp2 in game.legal_actions(s3):
            s4 = game.apply_action(s3, a_opp2)
            if game.is_terminal(s4) and game.winner(s4) == opp:
                opp_can_win_from_s3 = True
                break
        if not opp_can_win_from_s3:
            return False
    return True


def lose_in_two_if_wrong(game: Connect4Game, state: Connect4State) -> tuple[set[int], set[int]] | None:
    """
    Current player has at least one safe move; some moves allow opponent to force win in 2.
    Return (safe_moves, losing_moves) or None if no such split.
    """
    if game.is_terminal(state):
        return None
    us = game.current_player(state)
    opp = -us
    legal = list(game.legal_actions(state))
    safe: set[int] = set()
    losing: set[int] = set()
    for a1 in legal:
        s1 = game.apply_action(state, a1)
        if game.is_terminal(s1):
            if game.winner(s1) != us:
                safe.add(a1)
            continue
        opp_legal = list(game.legal_actions(s1))
        opp_can_force_win = False
        for a_opp in opp_legal:
            s2 = game.apply_action(s1, a_opp)
            if game.is_terminal(s2) and game.winner(s2) == opp:
                opp_can_force_win = True
                break
            if game.is_terminal(s2):
                continue
            if _opp_force_win_in_two_from_our_turn(game, s2, opp):
                opp_can_force_win = True
                break
        if opp_can_force_win:
            losing.add(a1)
        else:
            safe.add(a1)
    if safe and losing:
        return (safe, losing)
    return None


def win_in_two_moves(game: Connect4Game, state: Connect4State) -> set[int] | None:
    """
    Current player can force win in 2 full moves (our move, their move, our move = we win).
    Return set of winning first moves, or None if no such move exists.
    """
    if game.is_terminal(state):
        return None
    us = game.current_player(state)
    legal = list(game.legal_actions(state))
    winning_first: set[int] = set()
    for a1 in legal:
        s1 = game.apply_action(state, a1)
        if game.is_terminal(s1):
            if game.winner(s1) == us:
                winning_first.add(a1)
            continue
        opp_legal = list(game.legal_actions(s1))
        all_responses_we_win = True
        for a_opp in opp_legal:
            s2 = game.apply_action(s1, a_opp)
            if game.is_terminal(s2):
                w = game.winner(s2)
                if w is not None and w != 0 and w != us:
                    all_responses_we_win = False
                    break
                continue
            our_legal2 = list(game.legal_actions(s2))
            we_can_win = False
            for a2 in our_legal2:
                s3 = game.apply_action(s2, a2)
                if game.is_terminal(s3) and game.winner(s3) == us:
                    we_can_win = True
                    break
            if not we_can_win:
                all_responses_we_win = False
                break
        if all_responses_we_win:
            winning_first.add(a1)
    return winning_first if winning_first else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Find win-in-2 positions where depth-3 fails")
    parser.add_argument("--num-games", type=int, default=5000, help="Random games to play to collect positions")
    parser.add_argument("--min-pieces", type=int, default=4, help="Min pieces on board to consider position")
    parser.add_argument("--max-positions", type=int, default=500, help="Max win-in-2 positions to test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None, help="Output file for failing positions (npz)")
    parser.add_argument("--alpha-beta", action="store_true", help="Use alpha-beta (to find AB-specific failures)")
    parser.add_argument(
        "--only-win-in-2",
        action="store_true",
        help="Only test win-in-2 positions (depth-3 should see these; avoid testing lose-in-2-avoid which needs depth 4+)",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    game = Connect4Game(rows=6, cols=7)

    win_in_two_positions: list[tuple[Connect4State, set[int]]] = []
    lose_in_two_positions: list[tuple[Connect4State, set[int], set[int]]] = []  # state, safe, losing
    seen_hashes: set[str] = set()

    for g in range(args.num_games):
        state = game.initial_state()
        move_count = 0
        while not game.is_terminal(state):
            legal = list(game.legal_actions(state))
            if not legal:
                break
            action = rng.choice(legal)
            state = game.apply_action(state, action)
            move_count += 1
            if move_count >= args.min_pieces and not game.is_terminal(state):
                key = (state.board.tobytes(), state.current_player_index)
                if key in seen_hashes:
                    pass
                else:
                    seen_hashes.add(key)
                    wins = win_in_two_moves(game, state)
                    if wins is not None:
                        state_copy = Connect4State(
                            board=state.board.copy(),
                            current_player_index=state.current_player_index,
                            winner=state.winner,
                            done=state.done,
                            last_move=state.last_move,
                        )
                        win_in_two_positions.append((state_copy, wins))
                    if not args.only_win_in_2:
                        lose_info = lose_in_two_if_wrong(game, state)
                        if lose_info is not None:
                            safe, losing = lose_info
                            state_copy = Connect4State(
                                board=state.board.copy(),
                                current_player_index=state.current_player_index,
                                winner=state.winner,
                                done=state.done,
                                last_move=state.last_move,
                            )
                            lose_in_two_positions.append((state_copy, safe, losing))
                if len(win_in_two_positions) >= args.max_positions and (
                    args.only_win_in_2 or len(lose_in_two_positions) >= args.max_positions
                ):
                    break
        if len(win_in_two_positions) >= args.max_positions and (
            args.only_win_in_2 or len(lose_in_two_positions) >= args.max_positions
        ):
            break

    print(f"Collected {len(win_in_two_positions)} win-in-2, {len(lose_in_two_positions)} lose-in-2-if-wrong positions", file=sys.stderr)

    policy = make_connect4_heuristic_minimax_policy(
        game, depth=3, heuristic="trivial", use_alpha_beta=args.alpha_beta, random_tiebreak=False
    )

    failures: list[tuple[np.ndarray, int, int, set[int], str]] = []  # board, cpi, chosen, good_moves, kind

    for state, winning_moves in win_in_two_positions:
        chosen = policy.select_action(game, state)
        if chosen not in winning_moves:
            failures.append((
                state.board.copy(),
                state.current_player_index,
                int(chosen),
                winning_moves,
                "win_in_2",
            ))

    for state, safe_moves, losing_moves in lose_in_two_positions:
        chosen = policy.select_action(game, state)
        if chosen in losing_moves:
            failures.append((
                state.board.copy(),
                state.current_player_index,
                int(chosen),
                safe_moves,
                "lose_in_2_avoid",
            ))

    total_tested = len(win_in_two_positions) + len(lose_in_two_positions)
    print(f"Depth-3 failures: {len(failures)} / {total_tested} (win-in-2: {len(win_in_two_positions)}, lose-in-2-avoid: {len(lose_in_two_positions)})")
    if failures:
        print("First 20 failing positions (board, current_player_index, chosen_move, good_moves, kind):")
        for i, (board, cpi, chosen, good_moves, kind) in enumerate(failures[:20]):
            print(f"  {i+1}. kind={kind}, current_player_index={cpi}, chosen={chosen}, good_moves={good_moves}")
            print(board)
            print()
    if args.out and failures:
        np.savez(
            args.out,
            boards=np.stack([f[0] for f in failures]),
            current_player_indices=np.array([f[1] for f in failures], dtype=np.int32),
            chosen_moves=np.array([f[2] for f in failures], dtype=np.int32),
            good_moves=[np.array(list(f[3]), dtype=np.int32) for f in failures],
            kinds=np.array([f[4] for f in failures], dtype=object),
        )
        print(f"Saved {len(failures)} failures to {args.out}")


if __name__ == "__main__":
    main()
