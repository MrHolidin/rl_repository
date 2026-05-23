#!/usr/bin/env python3
"""Generate and render BGLike JSONL replays."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.envs  # noqa: F401
from src.agents.random_agent import RandomAgent
from src.envs.bglike.heuristic_bots import make_heuristic_agent
from src.envs.bglike.lobby_env import BGLobbyEnv
from src.envs.bglike.replay import ReplayHeuristicEnvBridge, attach_replay, close_replay
from src.envs.bglike.replay_render import render_jsonl_file
from src.envs.bglike.seat_config import lobby_from_learned_seats
from src.evaluation.eval_checkpoints import find_checkpoints, load_training_agent_checkpoint


def _build_env(
    *,
    seed: int,
    bot: str,
    record_seats: frozenset[int] | None,
    replay_path: Path,
    replay_meta: dict,
    agent_by_seat: dict | None = None,
) -> BGLobbyEnv:
    learned = tuple(range(8))
    if agent_by_seat is None:
        if bot == "random":
            agent_by_seat = {s: RandomAgent(seed=seed + 100 + s) for s in learned}
        else:
            agent_by_seat = {s: make_heuristic_agent(bot, seed=seed + 100 + s) for s in learned}
    configs = lobby_from_learned_seats(learned, agent_by_seat=agent_by_seat)
    env = BGLobbyEnv(
        configs,
        learned_seats=learned,
        training_seats=(0,),
        seed=seed,
    )
    attach_replay(
        env,
        replay_path,
        {"bot": bot, "seed": seed, **replay_meta},
        record_seats=record_seats,
    )
    if bot != "random" and agent_by_seat is None:
        bridge = ReplayHeuristicEnvBridge(env)
        for agent in agent_by_seat.values():
            if hasattr(agent, "set_env"):
                agent.set_env(bridge)
    return env


def cmd_generate(args: argparse.Namespace) -> None:
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    env = _build_env(
        seed=args.seed,
        bot=args.bot,
        record_seats=None if not args.no_shop_frames else frozenset(),
        replay_path=out,
        replay_meta={"episodes": args.episodes},
    )
    try:
        for ep in range(args.episodes):
            env.reset(seed=args.seed + ep)
            env.drain_until_lobby_done(deterministic=args.deterministic)
    finally:
        close_replay(env)
    print(f"Wrote {out}", flush=True)
    if args.render_txt:
        txt = render_jsonl_file(out, extended=args.extended)
        txt_path = args.render_txt.resolve()
        txt_path.write_text(txt, encoding="utf-8")
        print(f"Wrote {txt_path}", flush=True)


def cmd_render(args: argparse.Namespace) -> None:
    text = render_jsonl_file(args.in_path.resolve(), extended=args.extended)
    if args.out:
        args.out.resolve().write_text(text, encoding="utf-8")
        print(f"Wrote {args.out.resolve()}", flush=True)
    else:
        sys.stdout.write(text)


def cmd_checkpoints(args: argparse.Namespace) -> None:
    ck_dir = args.checkpoint_dir.resolve()
    found = find_checkpoints(ck_dir, prefix=args.prefix)
    if not found:
        raise SystemExit(f"No checkpoints in {ck_dir} (prefix={args.prefix!r})")
    selected = found[-args.last :]
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using {len(selected)} checkpoints from {ck_dir}", flush=True)

    for i, (ck_path, step) in enumerate(selected):
        agent = load_training_agent_checkpoint(
            ck_path,
            device=args.device,
            seed=args.seed + i,
        )
        if hasattr(agent, "eval"):
            agent.eval()
        jpath = out_dir / f"{ck_path.stem}.jsonl"
        tpath = out_dir / f"{ck_path.stem}.txt"
        meta = {
            "checkpoint": ck_path.name,
            "step": step,
            "mode": args.mode,
            "seed": args.seed + i,
        }
        if args.mode == "self_play":
            agents = {s: agent for s in range(8)}
        else:
            agents = {0: agent}
            for s in range(1, 8):
                agents[s] = make_heuristic_agent(args.opponent, seed=args.seed + 1000 + i + s)
        env = _build_env(
            seed=args.seed + i,
            bot=args.opponent if args.mode != "self_play" else "checkpoint",
            record_seats=None if not args.no_shop_frames else frozenset(),
            replay_path=jpath,
            replay_meta=meta,
            agent_by_seat=agents,
        )
        if args.mode != "self_play":
            bridge = ReplayHeuristicEnvBridge(env)
            for seat, seat_agent in agents.items():
                if seat != 0 and hasattr(seat_agent, "set_env"):
                    seat_agent.set_env(bridge)
        try:
            env.reset(seed=args.seed + i)
            env.drain_until_lobby_done(deterministic=True)
        finally:
            close_replay(env)
        txt = render_jsonl_file(jpath, extended=False)
        tpath.write_text(txt, encoding="utf-8")
        print(f"Wrote {jpath.name} + {tpath.name}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Run full 8-player lobby and write JSONL")
    gen.add_argument("--out", type=Path, required=True, help="Output .jsonl path")
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--episodes", type=int, default=1)
    gen.add_argument(
        "--bot",
        type=str,
        default="structured",
        help="Heuristic bot name for all seats, or 'random'",
    )
    gen.add_argument(
        "--no-shop-frames",
        action="store_true",
        help="Skip shop-action frames (combat/elimination/lobby end still recorded)",
    )
    gen.add_argument("--deterministic", action="store_true", default=True)
    gen.add_argument("--render-txt", type=Path, default=None, help="Also write .txt render")
    gen.add_argument("--extended", action="store_true", help="Extended render if --render-txt")
    gen.set_defaults(func=cmd_generate)

    ren = sub.add_parser("render", help="Render JSONL to human-readable text")
    ren.add_argument("in_path", type=Path, help="Input .jsonl")
    ren.add_argument("--out", type=Path, default=None, help="Output .txt (stdout if omitted)")
    ren.add_argument("--extended", action="store_true", help="Board/hand/shop after every action")
    ren.set_defaults(func=cmd_render)

    ck = sub.add_parser("checkpoints", help="Generate replays for latest training checkpoints")
    ck.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=REPO_ROOT / "runs/bglike/dist_ppo_004/checkpoints",
    )
    ck.add_argument("--prefix", type=str, default="dist_bglike_ppo")
    ck.add_argument("--last", type=int, default=8, help="Use N latest checkpoints")
    ck.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "runs/bglike/dist_ppo_004/replays",
    )
    ck.add_argument("--seed", type=int, default=13)
    ck.add_argument("--device", type=str, default="cpu")
    ck.add_argument(
        "--mode",
        choices=("self_play", "vs_heuristic"),
        default="self_play",
        help="self_play: same checkpoint on all 8 seats; vs_heuristic: seat0=ckpt, rest=bot",
    )
    ck.add_argument("--opponent", type=str, default="structured")
    ck.add_argument("--no-shop-frames", action="store_true")
    ck.set_defaults(func=cmd_checkpoints)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
