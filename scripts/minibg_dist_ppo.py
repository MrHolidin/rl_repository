#!/usr/bin/env python3
"""Distributed PPO training for MiniBG: workers collect (CPU), host updates (GPU).

Examples
--------
# From config (CLI flags override config values):
python scripts/minibg_dist_ppo.py \\
    --config configs/minibg/ppo_structured_dist.yaml \\
    --checkpoint checkpoints/minibg_ppo_structured_50000.pt \\
    --run-dir runs/dist_ppo_001

# Pure CLI:
python scripts/minibg_dist_ppo.py \\
    --checkpoint checkpoints/minibg_ppo_structured_50000.pt \\
    --run-dir runs/dist_ppo_001 \\
    --workers 12 --rollout-steps 16384 --total-steps 20_000_000 --host-device cuda
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import src.envs  # noqa: F401 — register game IDs

from src.training.callbacks.metrics_file import MetricsFileCallback
from src.training.callbacks.status_file import StatusFileCallback
from src.training.distributed_trainer import (
    DistributedCheckpointCallback,
    DistributedTrainer,
)
from src.training.metrics_presets import resolve_metrics_csv_fieldnames


def _load_config(path: Path) -> Dict[str, Any]:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _parse_scripted_dist(s: str) -> Dict[str, float]:
    dist: Dict[str, float] = {}
    for part in s.split(","):
        part = part.strip()
        if ":" in part:
            k, v = part.split(":", 1)
            dist[k.strip()] = float(v.strip())
    return dist or {"random": 1.0}


def _scripted_dist_from_yaml(cfg: Dict[str, Any]) -> Dict[str, float]:
    raw = cfg.get("distributed", {}).get("opponent", {}).get("scripted_distribution", {})
    return {str(k): float(v) for k, v in raw.items()} if raw else {"random": 1.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=None, help="Path to YAML config (ppo_structured_dist.yaml)")
    ap.add_argument("--checkpoint", type=Path, default=None, help="Starting checkpoint (.pt) — overrides config")
    ap.add_argument("--run-dir", type=Path, default=None, help="Output directory — overrides config")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--rollout-steps", type=int, default=None)
    ap.add_argument("--ppo-epochs", type=int, default=None)
    ap.add_argument("--minibatch-size", type=int, default=None)
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--host-device", type=str, default=None)
    ap.add_argument("--worker-device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--battle-damage-shaping", type=float, default=None)
    ap.add_argument("--checkpoint-interval", type=int, default=None)
    ap.add_argument("--current-fraction", type=float, default=None)
    ap.add_argument("--scripted-fraction", type=float, default=None)
    ap.add_argument("--scripted-distribution", type=str, default=None,
                    help="Comma-separated key:weight pairs, e.g. 't1_random:0.5,t_up_random:0.5'")
    ap.add_argument("--max-pool-size", type=int, default=None)
    args = ap.parse_args()

    # --- Load YAML config (base defaults) ---
    cfg: Dict[str, Any] = {}
    if args.config is not None:
        cfg = _load_config(args.config.resolve())

    dist_cfg = cfg.get("distributed", {})
    opp_cfg = dist_cfg.get("opponent", {})
    agent_cfg = cfg.get("agent", {}).get("params", {})
    game_cfg = cfg.get("game", {}).get("params", {})

    def _get(cli_val: Optional[Any], *cfg_keys: str, default: Any) -> Any:
        if cli_val is not None:
            return cli_val
        node: Any = cfg
        for k in cfg_keys:
            if not isinstance(node, dict):
                return default
            node = node.get(k, default)
            if node is default:
                return default
        return node

    workers: int           = _get(args.workers,           "distributed", "workers",             default=4)
    rollout_steps: int     = _get(args.rollout_steps,     "agent", "params", "rollout_steps",   default=8096)
    ppo_epochs: int        = _get(args.ppo_epochs,        "agent", "params", "ppo_epochs",      default=4)
    minibatch_size: int    = _get(args.minibatch_size,    "agent", "params", "minibatch_size",  default=512)
    total_steps: int       = _get(args.total_steps,       "distributed", "total_steps",         default=2_000_000)
    host_device: str       = _get(args.host_device,       "distributed", "host_device",         default="cuda")
    worker_device: str     = _get(args.worker_device,     "distributed", "worker_device",       default="cpu")
    seed: int              = _get(args.seed,               "seed",                               default=42)
    battle_shaping: float  = _get(args.battle_damage_shaping, "game", "params", "battle_damage_shaping", default=0.06)
    ckpt_interval: int     = _get(args.checkpoint_interval, "distributed", "checkpoint_interval", default=50_000)
    current_frac: float    = _get(args.current_fraction,  "distributed", "opponent", "current_fraction",  default=0.4)
    scripted_frac: float   = _get(args.scripted_fraction, "distributed", "opponent", "scripted_fraction", default=0.2)
    max_pool: int          = _get(args.max_pool_size,     "distributed", "opponent", "max_pool_size",      default=20)
    ema_beta: float        = _get(None,                   "distributed", "opponent", "ema_beta",           default=0.05)

    if args.scripted_distribution is not None:
        scripted_dist = _parse_scripted_dist(args.scripted_distribution)
    else:
        scripted_dist = _scripted_dist_from_yaml(cfg) if cfg else {"t1_random": 0.5, "t_up_random": 0.5}

    # --- Resolve checkpoint and run-dir ---
    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required")
    if args.run_dir is None:
        raise SystemExit("--run-dir is required")

    ck = args.checkpoint.resolve()
    if not ck.is_file():
        raise SystemExit(f"checkpoint not found: {ck}")

    import torch
    if host_device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA not available for --host-device; use --host-device cpu")

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "stop").unlink(missing_ok=True)

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    fieldnames = resolve_metrics_csv_fieldnames("ppo")

    callbacks = [
        StatusFileCallback(run_dir, interval=1, total_steps=total_steps),
        DistributedCheckpointCallback(ckpt_dir, interval=ckpt_interval, prefix="dist_minibg_ppo"),
        MetricsFileCallback(run_dir, interval=1, fieldnames=fieldnames),
    ]

    trainer = DistributedTrainer(
        ck,
        workers=workers,
        rollout_steps=rollout_steps,
        ppo_epochs=ppo_epochs,
        minibatch_size=minibatch_size,
        host_device=host_device,
        worker_device=worker_device,
        seed=seed,
        game_kwargs={"battle_damage_shaping": battle_shaping},
        callbacks=callbacks,
        current_fraction=current_frac,
        past_fraction=max(0.0, 1.0 - current_frac - scripted_frac),
        scripted_distribution=scripted_dist,
        max_pool_size=max_pool,
        ema_beta=ema_beta,
    )

    def _graceful_stop(sig: int, frame: object) -> None:
        print(f"\n[dist_ppo] signal {sig} — stopping after current round", flush=True)
        trainer.stop_training = True

    signal.signal(signal.SIGINT, _graceful_stop)
    signal.signal(signal.SIGTERM, _graceful_stop)

    print(
        f"[dist_ppo] checkpoint={ck.name}  workers={workers}  "
        f"rollout_steps={rollout_steps}  ppo_epochs={ppo_epochs}  minibatch={minibatch_size}\n"
        f"  host={host_device}  worker={worker_device}  "
        f"total_steps={total_steps:,}  run_dir={run_dir}\n"
        f"  opponent: current={current_frac}  scripted={scripted_frac}  "
        f"scripted_dist={scripted_dist}  pool_size={max_pool}",
        flush=True,
    )

    trainer.train(total_steps)
    print("[dist_ppo] training complete", flush=True)


if __name__ == "__main__":
    main()
