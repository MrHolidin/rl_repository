"""Distributed PPO training pipeline — mirrors run.py, adds distributed: config block."""

from __future__ import annotations

import signal
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Union

from src.agents.ppo_structured_minibg_agent import MiniBGPPOStructuredAgent
from src.config import load_config
from src.features.action_space import DiscreteActionSpace
from src.registry import make_agent, make_game
from src.training.callbacks import MetricsFileCallback, StatusFileCallback
from src.training.distributed_trainer import (
    DistributedCheckpointCallback,
    DistributedTrainer,
    _apply_ppo_hparams,
)
from src.training.meta import collect_meta, write_meta_json
from src.training.run import (
    _build_metrics_file_callback,
    _build_scripted_spec,
    _remove_pid,
    _resolve_device,
    _set_seed,
    _write_pid,
)

_current_trainer: Optional[DistributedTrainer] = None
_original_sigterm_handler: Optional[Callable] = None
_original_sigint_handler: Optional[Callable] = None


def _sigterm_handler(signum: int, frame: Any) -> None:
    if _current_trainer is not None:
        _current_trainer.stop_training = True


def _install_signal_handlers() -> None:
    global _original_sigterm_handler, _original_sigint_handler
    try:
        _original_sigterm_handler = signal.signal(signal.SIGTERM, _sigterm_handler)
        _original_sigint_handler = signal.signal(signal.SIGINT, _sigterm_handler)
    except (ValueError, OSError):
        pass


def _restore_signal_handlers() -> None:
    try:
        if _original_sigterm_handler is not None:
            signal.signal(signal.SIGTERM, _original_sigterm_handler)
        if _original_sigint_handler is not None:
            signal.signal(signal.SIGINT, _original_sigint_handler)
    except (ValueError, OSError):
        pass


def run_distributed(
    config_path: Union[str, Path],
    run_dir: Union[str, Path],
    *,
    command: Optional[str] = None,
) -> None:
    """Run distributed PPO training from a YAML config.

    The config uses the same schema as single-process training plus a
    ``distributed:`` block that specifies workers, worker_device, and an
    optional starting checkpoint path.
    """
    global _current_trainer

    config_path = Path(config_path)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    _write_pid(run_dir)

    stop_path = run_dir / "stop"
    if stop_path.exists():
        try:
            stop_path.unlink()
        except OSError:
            pass

    (run_dir / "config.yaml").write_text(config_path.read_text())

    app_cfg = load_config(config_path)
    dist_cfg = app_cfg.distributed
    if dist_cfg is None:
        raise SystemExit(
            "Config has no 'distributed:' section. "
            "Use src.cli.train for single-process training."
        )

    if app_cfg.seed is not None:
        _set_seed(app_cfg.seed)
    seed: int = app_cfg.seed if app_cfg.seed is not None else 42

    # --- Agent creation (identical to run.py) ---
    game_params = dict(app_cfg.game.params or {})
    game_params.pop("control_driver", None)
    base_env = make_game(app_cfg.game.id, **game_params)
    legal_mask = getattr(base_env, "legal_actions_mask", None)
    if legal_mask is None:
        raise ValueError("Environment must expose legal_actions_mask.")
    num_actions = int(len(legal_mask))

    agent_params = dict(app_cfg.agent.params)
    agent_params.setdefault("num_actions", num_actions)
    agent_params.setdefault("action_space", DiscreteActionSpace(num_actions))
    obs_builder = getattr(base_env, "observation_builder", None)
    if obs_builder is not None:
        agent_params.setdefault("observation_shape", obs_builder.observation_shape)
        agent_params.setdefault(
            "observation_type", getattr(obs_builder, "observation_type", "board")
        )
    device = _resolve_device(agent_params.get("device"))
    agent_params["device"] = device

    try:
        import torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit("CUDA not available; set agent.params.device: cpu in config.")
    except ImportError:
        pass

    # Load from checkpoint or create fresh; either way workers get a file path.
    ck_path = Path(dist_cfg.checkpoint).resolve() if dist_cfg.checkpoint else None
    if ck_path is not None and ck_path.is_file():
        agent = MiniBGPPOStructuredAgent.load(str(ck_path), device=device, seed=seed)
        worker_ckpt_path = ck_path
        print(f"[dist_ppo] loading checkpoint: {ck_path.name}", flush=True)
    else:
        agent = make_agent(app_cfg.agent.id, **agent_params)
        worker_ckpt_path = ckpt_dir / "init.pt"
        agent.save(str(worker_ckpt_path))
        print(f"[dist_ppo] fresh agent saved to {worker_ckpt_path}", flush=True)

    rollout_steps = int(agent_params.get("rollout_steps", 8096))
    ppo_epochs = int(agent_params.get("ppo_epochs", 4))
    minibatch_size = int(agent_params.get("minibatch_size", 512))
    _apply_ppo_hparams(
        agent, rollout_steps=rollout_steps, ppo_epochs=ppo_epochs, minibatch_size=minibatch_size
    )
    agent.train()

    # --- Opponent params (from train.opponent_sampler, same as single-process) ---
    sp = dict((app_cfg.train.opponent_sampler.params.get("self_play") or {}))
    current_fraction = float(sp.get("current_self_fraction", 0.4))
    past_fraction = float(sp.get("past_self_fraction", 0.4))
    start_episode = int(sp.get("start_episode", 0))
    max_pool_size = int(sp.get("max_frozen_agents", 20))
    ema_beta = float(sp.get("frozen_ema_beta", 0.05))

    scripted_spec = _build_scripted_spec(app_cfg.game.id, app_cfg.train.opponent_sampler.params)
    scripted_distribution = dict(scripted_spec.distribution)

    # --- Callbacks (same config keys as single-process) ---
    total_steps = app_cfg.train.total_steps

    ckpt_cb_cfg = next(
        (c for c in app_cfg.train.callbacks if c.type.lower() == "checkpoint" and c.enabled),
        None,
    )
    ckpt_interval = int((ckpt_cb_cfg.params.get("interval") if ckpt_cb_cfg else None) or 50_000)
    ckpt_prefix = str((ckpt_cb_cfg.params.get("prefix") if ckpt_cb_cfg else None) or "checkpoint")

    metrics_cb_cfg = next(
        (c for c in app_cfg.train.callbacks if c.type.lower() == "metrics_file" and c.enabled),
        None,
    )
    metrics_cb = (
        _build_metrics_file_callback(run_dir, metrics_cb_cfg, agent_id=app_cfg.agent.id)
        if metrics_cb_cfg is not None
        else MetricsFileCallback(run_dir, interval=1)
    )

    callbacks = [
        StatusFileCallback(run_dir, interval=1, total_steps=total_steps),
        DistributedCheckpointCallback(ckpt_dir, interval=ckpt_interval, prefix=ckpt_prefix),
    ]
    if metrics_cb is not None:
        callbacks.append(metrics_cb)

    meta = collect_meta(
        config_path=config_path,
        command=command or " ".join(sys.argv),
        seed=seed,
        device=device,
    )
    write_meta_json(run_dir / "meta.json", meta)

    trainer = DistributedTrainer(
        agent,
        worker_ckpt_path,
        workers=dist_cfg.workers,
        rollout_steps=rollout_steps,
        ppo_epochs=ppo_epochs,
        minibatch_size=minibatch_size,
        worker_device=dist_cfg.worker_device,
        seed=seed,
        game_kwargs=dict(game_params),
        callbacks=callbacks,
        current_fraction=current_fraction,
        past_fraction=past_fraction,
        scripted_distribution=scripted_distribution,
        max_pool_size=max_pool_size,
        ema_beta=ema_beta,
        start_episode=start_episode,
        run_dir=str(run_dir),
    )

    _current_trainer = trainer
    _install_signal_handlers()

    try:
        trainer.train(total_steps)
    finally:
        _current_trainer = None
        _restore_signal_handlers()
        _remove_pid(run_dir)
