#!/usr/bin/env python3
"""Benchmark workers collect rollout (whole games, min steps) → host PPO → broadcast weights."""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import pickle
import sys
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import src.envs  # noqa: F401

from src.agents.ppo_structured_minibg_agent import (
    INFO_STRUCT_LEGAL,
    INFO_STRUCT_NEXT_LEGAL,
    MiniBGPPOStructuredAgent,
    StructuredMiniBGRolloutBuffer,
)
from src.envs import RewardConfig
from src.registry import make_game
from src.training.agent_perspective_env import AgentPerspectiveEnv, make_minibg_shaping_fn
from src.training.opponent_sampler import OpponentSampler
from src.training.trainer import Transition


@dataclass
class RoundTiming:
    play_s: float
    upload_s: float
    host_s: float
    download_s: float

    @property
    def total_s(self) -> float:
        return self.play_s + self.upload_s + self.host_s + self.download_s


class FixedOpponentSampler(OpponentSampler):
    def __init__(self, opponent: Any) -> None:
        self._opponent = opponent

    def prepare(self, episode_index: int) -> None:
        pass

    def sample(self) -> Any:
        return self._opponent

    def on_episode_end(self, episode_index: int, info: Optional[dict] = None) -> None:
        pass


def _apply_ppo_hparams(
    agent: MiniBGPPOStructuredAgent,
    *,
    rollout_steps: int,
    ppo_epochs: int,
    minibatch_size: int,
) -> None:
    agent.rollout_steps = int(rollout_steps)
    agent.ppo_epochs = int(ppo_epochs)
    agent.minibatch_size = int(minibatch_size)


def _buffer_to_payload(buf: StructuredMiniBGRolloutBuffer) -> bytes:
    return pickle.dumps(buf, protocol=pickle.HIGHEST_PROTOCOL)


def _payload_to_buffer(data: bytes) -> StructuredMiniBGRolloutBuffer:
    return pickle.loads(data)


def _merge_buffers(buffers: List[StructuredMiniBGRolloutBuffer]) -> StructuredMiniBGRolloutBuffer:
    out = StructuredMiniBGRolloutBuffer()
    for buf in buffers:
        out.obs.extend(buf.obs)
        out.legal_lists.extend(buf.legal_lists)
        out.action_indices.extend(buf.action_indices)
        out.complete_turn.extend(buf.complete_turn)
        out.occupied_masks.extend(buf.occupied_masks)
        out.order_picks.extend(buf.order_picks)
        out.rewards.extend(buf.rewards)
        out.dones.extend(buf.dones)
        out.values.extend(buf.values)
        out.log_probs.extend(buf.log_probs)
        out.next_obs.extend(buf.next_obs)
        out.next_legal_lists.extend(buf.next_legal_lists)
    return out


def _state_dict_bytes(agent: MiniBGPPOStructuredAgent, *, map_cpu: bool = False) -> bytes:
    import torch

    sd = agent.policy_net.state_dict()
    if map_cpu:
        sd = {k: v.detach().cpu() for k, v in sd.items()}
    bio = BytesIO()
    torch.save(sd, bio)
    return bio.getvalue()


def _load_state_dict_bytes(agent: MiniBGPPOStructuredAgent, payload: bytes) -> None:
    import torch

    bio = BytesIO(payload)
    sd = torch.load(bio, map_location=agent.device, weights_only=True)
    agent.policy_net.load_state_dict(sd)


def _collect_until_steps(
    agent: MiniBGPPOStructuredAgent,
    opponent: MiniBGPPOStructuredAgent,
    *,
    min_steps: int,
    seed: int,
    mg: dict,
) -> Tuple[int, int, StructuredMiniBGRolloutBuffer]:
    import torch

    torch.set_num_threads(1)
    agent.train()
    opponent.eval()
    if hasattr(opponent, "epsilon"):
        opponent.epsilon = 0.0

    base = make_game("minibg", reward_config=RewardConfig(), **mg)
    shaping = make_minibg_shaping_fn(float(mg.get("battle_damage_shaping", 0.0)))
    env = AgentPerspectiveEnv(
        base,
        FixedOpponentSampler(opponent),
        agent_first_probability=0.5,
        shaping_fn=shaping,
    )

    n_games = 0
    game_idx = 0
    while len(agent.rollout_buffer) < min_steps:
        obs = env.reset()
        while not env.done:
            legal_list = env.legal_structured_actions()
            struct_act, board_perm, idx = agent.act_structured(
                obs, legal_list, env, deterministic=False
            )
            step = env.step_structured(struct_act, board_perm=board_perm)
            next_sl = [] if step.done else list(env.legal_structured_actions())
            transition = Transition(
                obs=obs,
                action=idx,
                reward=float(step.reward),
                next_obs=step.obs,
                terminated=step.terminated,
                truncated=step.truncated,
                info={
                    **(step.info if isinstance(step.info, dict) else {}),
                    INFO_STRUCT_LEGAL: legal_list,
                    INFO_STRUCT_NEXT_LEGAL: next_sl,
                },
                legal_mask=None,
                next_legal_mask=None,
            )
            agent.observe(transition)
            obs = step.obs
        env.notify_episode_end(step.info if isinstance(step.info, dict) else {})
        n_games += 1
        game_idx += 1

    buf = agent.rollout_buffer
    n_steps = len(buf)
    agent.rollout_buffer = StructuredMiniBGRolloutBuffer()
    return n_games, n_steps, buf


def _worker_main(
    worker_id: int,
    ck_path: str,
    min_steps: int,
    rollout_steps: int,
    ppo_epochs: int,
    minibatch_size: int,
    seed: int,
    mg: dict,
    device: str,
    cmd_conn: Any,
) -> None:
    import torch

    torch.set_num_threads(1)
    ck = Path(ck_path)
    agent = MiniBGPPOStructuredAgent.load(str(ck), device=device, seed=seed + worker_id)
    opponent = MiniBGPPOStructuredAgent.load(str(ck), device=device, seed=seed + worker_id + 913_123)
    _apply_ppo_hparams(
        agent,
        rollout_steps=rollout_steps,
        ppo_epochs=ppo_epochs,
        minibatch_size=minibatch_size,
    )
    agent.train()

    while True:
        msg = cmd_conn.recv()
        if msg[0] == "load":
            _load_state_dict_bytes(agent, msg[1])
            _load_state_dict_bytes(opponent, msg[1])
        elif msg[0] == "play":
            t0 = time.perf_counter()
            ng, nsteps, buf = _collect_until_steps(
                agent,
                opponent,
                min_steps=min_steps,
                seed=seed + worker_id * 10_000 + msg[1] * 1_000_003,
                mg=mg,
            )
            play_s = time.perf_counter() - t0
            t1 = time.perf_counter()
            payload = _buffer_to_payload(buf)
            upload_s = time.perf_counter() - t1
            cmd_conn.send(("rollout", ng, nsteps, len(payload), play_s, upload_s, payload))
        elif msg[0] == "stop":
            break
        else:
            raise RuntimeError(f"unknown worker cmd: {msg[0]}")


def _host_sync(
    host_agent: MiniBGPPOStructuredAgent,
    payloads: List[bytes],
    *,
    rollout_steps: int,
) -> Tuple[bytes, int, Dict[str, float]]:
    buffers = [_payload_to_buffer(p) for p in payloads]
    merged = _merge_buffers(buffers)
    n_steps = len(merged)
    if n_steps < rollout_steps:
        raise RuntimeError(
            f"merged rollout {n_steps} < rollout_steps {rollout_steps}; "
            "increase per-worker min steps or worker count"
        )
    host_agent.rollout_buffer = merged
    metrics = host_agent.update()
    if "policy_loss" not in metrics:
        raise RuntimeError(f"host update did not train: {metrics!r}")
    return _state_dict_bytes(host_agent, map_cpu=True), n_steps, metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-steps-per-worker", type=int, default=1024)
    ap.add_argument("--rollout-steps", type=int, default=8096)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--minibatch-size", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument(
        "--host-device",
        type=str,
        default="cuda",
        help="Device for host PPO update (default: cuda).",
    )
    ap.add_argument(
        "--worker-device",
        type=str,
        default="cpu",
        help="Device for worker rollout collection (default: cpu).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--battle-damage-shaping", type=float, default=0.06)
    args = ap.parse_args()

    ck = args.checkpoint.resolve()
    if not ck.is_file():
        raise SystemExit(f"checkpoint not found: {ck}")

    import torch

    host_device = str(args.host_device)
    worker_device = str(args.worker_device)
    if host_device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA not available for --host-device")

    workers = int(args.workers)
    min_steps_user = int(args.min_steps_per_worker)
    rollout_steps = int(args.rollout_steps)
    ppo_epochs = int(args.ppo_epochs)
    minibatch_size = int(args.minibatch_size)
    rounds = int(args.rounds)

    per_worker_target = math.ceil(rollout_steps / workers)
    if min_steps_user > per_worker_target:
        print(
            f"warning: --min-steps-per-worker={min_steps_user} ignored for {workers} workers "
            f"(fair share {per_worker_target} => {workers * per_worker_target} total for "
            f"rollout_steps={rollout_steps})",
            flush=True,
        )
    elif min_steps_user < per_worker_target:
        per_worker_target = max(min_steps_user, per_worker_target)
    mg = {"battle_damage_shaping": float(args.battle_damage_shaping)}

    ctx = mp.get_context("spawn")
    conns: List[Any] = []
    procs: List[mp.Process] = []
    for w in range(workers):
        parent, child = ctx.Pipe(duplex=True)
        p = ctx.Process(
            target=_worker_main,
            args=(
                w,
                str(ck),
                per_worker_target,
                rollout_steps,
                ppo_epochs,
                minibatch_size,
                args.seed,
                mg,
                worker_device,
                child,
            ),
            daemon=True,
        )
        p.start()
        child.close()
        conns.append(parent)
        procs.append(p)

    host_agent = MiniBGPPOStructuredAgent.load(str(ck), device=host_device, seed=args.seed)
    _apply_ppo_hparams(
        host_agent,
        rollout_steps=rollout_steps,
        ppo_epochs=ppo_epochs,
        minibatch_size=minibatch_size,
    )
    host_agent.train()
    init_sd = _state_dict_bytes(host_agent, map_cpu=True)

    timings: List[RoundTiming] = []
    total_steps = 0
    total_games = 0
    total_upload_bytes = 0
    per_worker_steps: List[int] = []

    print(
        f"checkpoint={ck.name}  workers={workers}  rounds={rounds}\n"
        f"  worker_device={worker_device}  host_device={host_device}  "
        f"(host policy: {next(host_agent.policy_net.parameters()).device})\n"
        f"  rollout_steps={rollout_steps}  ppo_epochs={ppo_epochs}  minibatch_size={minibatch_size}\n"
        f"  steps/worker={per_worker_target}  (ceil({rollout_steps}/{workers}), "
        f"expected merged>={rollout_steps})",
        flush=True,
    )

    t_all = time.perf_counter()
    sd_bytes = init_sd
    for r in range(rounds):
        play_upload: List[Tuple[float, float, bytes, int, int, int]] = []
        for conn in conns:
            conn.send(("play", r))
        for conn in conns:
            tag, ng, nsteps, nbytes, play_s, upload_s, payload = conn.recv()
            assert tag == "rollout"
            play_upload.append((play_s, upload_s, payload, ng, nsteps))
            total_upload_bytes += nbytes
            total_games += ng
            per_worker_steps.append(nsteps)

        play_s = max(p[0] for p in play_upload)
        upload_s = max(p[1] for p in play_upload)
        payloads = [p[2] for p in play_upload]
        round_games = sum(p[3] for p in play_upload)

        t0 = time.perf_counter()
        sd_bytes, n_steps, metrics = _host_sync(
            host_agent, payloads, rollout_steps=rollout_steps
        )
        host_s = time.perf_counter() - t0
        total_steps += n_steps

        t0 = time.perf_counter()
        for conn in conns:
            conn.send(("load", sd_bytes))
        download_s = time.perf_counter() - t0

        timings.append(RoundTiming(play_s, upload_s, host_s, download_s))
        pl = metrics.get("policy_loss")
        print(
            f"round {r + 1}/{rounds}: merged_steps={n_steps}  games={round_games}  "
            f"worker_steps={per_worker_steps[-workers:]}  "
            f"play={play_s:.2f}s upload={upload_s:.2f}s host={host_s:.2f}s dl={download_s:.2f}s  "
            f"policy_loss={pl:.4f}",
            flush=True,
        )

    wall = time.perf_counter() - t_all
    for conn in conns:
        conn.send(("stop",))
    for p in procs:
        p.join(timeout=60)

    sync_overhead = sum(t.upload_s + t.host_s + t.download_s for t in timings)
    play_total = sum(t.play_s for t in timings)

    print()
    print("=== distributed sync benchmark ===")
    print(
        f"workers={workers}  rounds={rounds}  "
        f"rollout_steps={rollout_steps}  ppo_epochs={ppo_epochs}  minibatch={minibatch_size}"
    )
    print(f"per-worker target steps: {per_worker_target}")
    print(f"total games={total_games}  total transitions={total_steps}")
    print(f"rollout upload bytes (sum workers): {total_upload_bytes / 1e6:.1f} MB")
    print(f"state_dict size: {len(sd_bytes) / 1e6:.2f} MB")
    print(f"wall time: {wall:.2f}s  -> {total_steps / wall:.1f} steps/s  {total_games / wall:.2f} games/s")
    print(f"  play only (max worker/round, summed): {play_total:.2f}s")
    print(f"  sync (upload+host+download, summed): {sync_overhead:.2f}s")
    print(f"avg per round: {wall / rounds:.2f}s")


if __name__ == "__main__":
    main()
