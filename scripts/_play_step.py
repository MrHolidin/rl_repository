#!/usr/bin/env python3
"""Stateless single-step driver for play_bglike_human (one command per call).

Persists the full game between calls in a pickle so an external agent can play
turn-by-turn: read the printed state, decide one command, call again with --cmd.

    # start a game (no --cmd)
    python scripts/_play_step.py --session /tmp/bg.pkl --ckpt <ckpt>
    # then issue commands
    python scripts/_play_step.py --session /tmp/bg.pkl --cmd "buy 0"
    python scripts/_play_step.py --session /tmp/bg.pkl --cmd "end"
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

torch.set_num_threads(1)

import src.envs  # noqa: F401
from src.envs.bglike.lobby_env import BGLobbyEnv
from src.envs.bglike.placement import is_seat_finished, placement_for_seat
from src.envs.bglike.seat_config import lobby_from_learned_seats
from src.evaluation.eval_checkpoints import load_training_agent_checkpoint
from scripts.play_bglike_human import (
    BadCommand,
    NUM_PLAYERS,
    handle_command,
    render_combat,
    render_state,
)


def build_env(meta: dict):
    agent = load_training_agent_checkpoint(
        Path(meta["ckpt"]), device=meta["device"], seed=meta["seed"]
    )
    if hasattr(agent, "eval"):
        agent.eval()
    seat = meta["seat"]
    bots = tuple(s for s in range(NUM_PLAYERS) if s != seat)
    cfg = lobby_from_learned_seats(
        bots, agent_by_seat={s: agent for s in bots}, seed=meta["seed"]
    )
    env = BGLobbyEnv(
        cfg,
        learned_seats=bots,
        seed=meta["seed"],
        patch_dir=meta["patch_dir"],
        obs_kind=meta["obs_kind"],
        with_heroes=meta["with_heroes"],
    )
    return env


def restore(env: BGLobbyEnv, bundle: dict) -> None:
    env._state = bundle["state"]
    env._rl_pending = bundle["rl_pending"]
    env._last_battle_signed = bundle["last_battle_signed"]
    env._finished_training = bundle["finished_training"]
    env._rl_place_budget_pending = bundle["rl_place_budget_pending"]
    # Restore the game's RNG position in-place so shop rolls / combat continue
    # the same stream across calls (build_env re-seeds a fresh Generator). All
    # consumers (shop triggers, combat, hero assignment) share this one object.
    env.game._rng.bit_generator.state = bundle["rng_state"]


def snapshot(env: BGLobbyEnv) -> dict:
    return {
        "state": env._state,
        "rl_pending": env._rl_pending,
        "last_battle_signed": env._last_battle_signed,
        "finished_training": env._finished_training,
        "rl_place_budget_pending": env._rl_place_budget_pending,
        "rng_state": env.game._rng.bit_generator.state,
    }


def advance(env: BGLobbyEnv, meta: dict) -> None:
    """Auto-play bots until the human must act or the lobby ends; report combats."""
    seat = meta["seat"]
    while not env.lobby_done:
        cur = env.current_seat()
        if cur == seat and env._seat_can_act(seat):
            break
        if env._seat_can_act(cur):
            env.step_auto(cur, deterministic=meta["greedy"])
        else:
            break
        if env.state.combat_round != meta["prev_cr"]:
            render_combat(env, seat, meta["prev_hp"], set(meta["prev_elim"]))
            meta["prev_cr"] = env.state.combat_round
            meta["prev_hp"] = env.state.players[seat].health
            meta["prev_elim"] = sorted({s.seat for s in env.state.eliminated})
        if is_seat_finished(env.state, seat):
            break


def render_or_final(env: BGLobbyEnv, meta: dict) -> bool:
    seat = meta["seat"]
    if env.lobby_done or is_seat_finished(env.state, seat):
        place = placement_for_seat(env.state, seat)
        print("\n" + "=" * 72)
        if env.lobby_done and env.state.winner == seat:
            print("You WON the lobby!")
        print(f"GAME OVER. Final placement: {place} / {NUM_PLAYERS}  (1 = best)")
        return True
    render_state(env, seat)
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--cmd", type=str, default=None)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--patch-dir", type=str, default="data/bgcore/19_6_0_74257")
    ap.add_argument("--obs-kind", type=str, default="bglike_v5_heroes")
    ap.add_argument("--with-heroes", dest="with_heroes", action="store_true", default=True)
    ap.add_argument("--no-heroes", dest="with_heroes", action="store_false")
    ap.add_argument("--greedy", dest="greedy", action="store_true", default=True)
    ap.add_argument("--sample", dest="greedy", action="store_false")
    args = ap.parse_args()

    fresh = not args.session.exists()
    if fresh:
        if args.ckpt is None:
            raise SystemExit("--ckpt required to start a new session")
        meta = {
            "ckpt": str(args.ckpt.resolve()),
            "seat": args.seat,
            "seed": args.seed,
            "device": args.device,
            "patch_dir": args.patch_dir,
            "obs_kind": args.obs_kind,
            "with_heroes": args.with_heroes,
            "greedy": args.greedy,
        }
        env = build_env(meta)
        env.reset(seed=meta["seed"])
        meta["prev_cr"] = env.state.combat_round
        meta["prev_hp"] = env.state.players[meta["seat"]].health
        meta["prev_elim"] = sorted({s.seat for s in env.state.eliminated})
        print(f"NEW GAME -- you are seat {meta['seat']}, bots: {Path(meta['ckpt']).name}")
    else:
        with open(args.session, "rb") as f:
            saved = pickle.load(f)
        meta = saved["meta"]
        if saved.get("done"):
            raise SystemExit("session already finished; delete it to start over")
        env = build_env(meta)
        restore(env, saved["bundle"])
        if args.cmd:
            seat = meta["seat"]
            if not (env.current_seat() == seat and env._seat_can_act(seat)):
                raise SystemExit("not your turn right now")
            try:
                handle_command(env, seat, args.cmd)
                print(f"[ok] {args.cmd}")
            except BadCommand as e:
                print(f"[illegal] {args.cmd!r}: {e}")
            except KeyboardInterrupt:
                raise SystemExit("quit")

    advance(env, meta)
    done = render_or_final(env, meta)

    with open(args.session, "wb") as f:
        pickle.dump({"meta": meta, "bundle": snapshot(env), "done": done}, f)


if __name__ == "__main__":
    main()
