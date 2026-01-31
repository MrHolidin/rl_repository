"""Training pipeline: run_dir, config copy, meta, callbacks, graceful stop."""

from __future__ import annotations

import os
import random
import signal
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from src.config import AppConfig, CallbackConfig, load_config
from src.features.action_space import DiscreteActionSpace
from src.registry import make_agent, make_game
from src.training.meta import collect_meta, write_meta_json
from src.training.opponent_sampler import OpponentSampler, RandomOpponentSampler
from src.training.callbacks import (
    CheckpointCallback,
    EpsilonDecayCallback,
    LearningRateDecayCallback,
    MetricsFileCallback,
    StatusFileCallback,
    TrainerCallback,
)
from src.training.random_opening import RandomOpeningConfig
from src.training.trainer import StartPolicy, Trainer


# Global reference for signal handler
_current_trainer: Optional[Trainer] = None
_original_sigterm_handler: Optional[Callable] = None
_original_sigint_handler: Optional[Callable] = None


def _sigterm_handler(signum: int, frame: Any) -> None:
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    if _current_trainer is not None:
        _current_trainer.stop_training = True


def _install_signal_handlers() -> None:
    """Install signal handlers for graceful stop."""
    global _original_sigterm_handler, _original_sigint_handler
    try:
        _original_sigterm_handler = signal.signal(signal.SIGTERM, _sigterm_handler)
        _original_sigint_handler = signal.signal(signal.SIGINT, _sigterm_handler)
    except (ValueError, OSError):
        # signal handlers can only be set in main thread
        pass


def _restore_signal_handlers() -> None:
    """Restore original signal handlers."""
    try:
        if _original_sigterm_handler is not None:
            signal.signal(signal.SIGTERM, _original_sigterm_handler)
        if _original_sigint_handler is not None:
            signal.signal(signal.SIGINT, _original_sigint_handler)
    except (ValueError, OSError):
        pass


def _write_pid(run_dir: Path) -> None:
    """Write current PID to run_dir/pid."""
    (run_dir / "pid").write_text(str(os.getpid()))


def _remove_pid(run_dir: Path) -> None:
    """Remove pid file."""
    pid_path = run_dir / "pid"
    if pid_path.exists():
        try:
            pid_path.unlink()
        except OSError:
            pass


def _resolve_device(device: Optional[str]) -> str:
    if device is not None and device.lower() in ("cuda", "cpu"):
        if device.lower() == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
        return device.lower()
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _build_checkpoint_callback(
    run_dir: Path,
    callback_cfgs: List[CallbackConfig],
) -> TrainerCallback:
    output_dir = run_dir / "checkpoints"
    interval = 2000
    prefix = "model"
    for cb in callback_cfgs:
        if cb.type.lower() == "checkpoint" and cb.enabled:
            interval = int(cb.params.get("interval", interval))
            prefix = str(cb.params.get("prefix", prefix))
            break
    return CheckpointCallback(output_dir=output_dir, interval=interval, prefix=prefix)


def _build_epsilon_decay_callback(cb: CallbackConfig) -> Optional[TrainerCallback]:
    if not cb.enabled or cb.type.lower() != "epsilon_decay":
        return None
    p = cb.params
    every = str(p.get("every", "step")).strip().lower()
    if every not in ("step", "episode"):
        every = "step"
    return EpsilonDecayCallback(every=every)


def _build_metrics_file_callback(
    run_dir: Path,
    cb: CallbackConfig,
) -> Optional[TrainerCallback]:
    if not cb.enabled or cb.type.lower() != "metrics_file":
        return None
    p = cb.params
    interval = int(p.get("interval", 100))
    filename = str(p.get("filename", "metrics.csv"))
    return MetricsFileCallback(run_dir=run_dir, interval=interval, filename=filename)


def _build_lr_decay_callback(cb: CallbackConfig) -> Optional[TrainerCallback]:
    if not cb.enabled or cb.type.lower() != "lr_decay":
        return None
    p = cb.params
    interval_steps = int(p.get("interval_steps", 1000))
    decay_factor = float(p.get("decay_factor", 0.5))
    min_lr = p.get("min_lr")
    if min_lr is not None:
        min_lr = float(min_lr)
    optimizer_attr = str(p.get("optimizer_attr", "optimizer"))
    metric_key = str(p.get("metric_key", "learning_rate"))
    return LearningRateDecayCallback(
        interval_steps=interval_steps,
        decay_factor=decay_factor,
        min_lr=min_lr,
        optimizer_attr=optimizer_attr,
        metric_key=metric_key,
    )


def _resolve_start_policy(s: str) -> StartPolicy:
    v = (s or "random").strip().lower()
    if v == "agent_first":
        return StartPolicy.AGENT_FIRST
    if v == "opponent_first":
        return StartPolicy.OPPONENT_FIRST
    return StartPolicy.RANDOM


def _build_opponent_sampler(
    app_cfg: AppConfig,
    agent: Any,
    device: str,
) -> OpponentSampler:
    from src.training.opponent_sampler import OpponentPoolSampler
    from src.training.selfplay import OpponentPool, SelfPlayConfig

    cfg = app_cfg.train.opponent_sampler
    seed = app_cfg.seed

    if cfg.type.lower() == "random":
        use_seed = cfg.params.get("seed")
        if use_seed is None:
            use_seed = seed
        return RandomOpponentSampler(seed=use_seed)

    if cfg.type.lower() == "pool":
        if seed is None:
            raise ValueError("opponent_sampler type 'pool' requires config seed.")
        sp = cfg.params.get("self_play", {})
        save_every = int(sp.get("save_every", 1000))
        self_play_config = SelfPlayConfig(
            start_episode=int(sp.get("start_episode", 0)),
            current_self_fraction=float(sp.get("current_self_fraction", 0.3)),
            past_self_fraction=float(sp.get("past_self_fraction", 0.3)),
            max_frozen_agents=int(sp.get("max_frozen_agents", 10)),
            save_every=max(1, save_every),
        )
        heuristic_distribution = dict(cfg.params.get("heuristic_distribution") or {})
        if not heuristic_distribution:
            heuristic_distribution = {"random": 0.2, "heuristic": 0.5, "smart_heuristic": 0.3}
        pool = OpponentPool(
            device=device,
            seed=seed,
            self_play_config=self_play_config,
            heuristic_distribution=heuristic_distribution,
            current_agent=agent,
        )
        return OpponentPoolSampler(opponent_pool=pool)
    raise ValueError(f"Unknown opponent_sampler type: {cfg.type}")


def run(
    config_path: Union[str, Path],
    run_dir: Union[str, Path],
    *,
    command: Optional[str] = None,
    status_interval: int = 100,
) -> None:
    """
    Run training from a config file.

    Args:
        config_path: Path to YAML config.
        run_dir: Directory for outputs (config copy, meta.json, checkpoints, status.json).
        command: Command string for meta.json (defaults to sys.argv).
        status_interval: How often to update status.json (in steps).
    """
    global _current_trainer

    config_path = Path(config_path)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # Write PID for external monitoring/stopping
    _write_pid(run_dir)

    # Remove stale stop file if present
    stop_path = run_dir / "stop"
    if stop_path.exists():
        try:
            stop_path.unlink()
        except OSError:
            pass

    raw_config = config_path.read_text()
    (run_dir / "config.yaml").write_text(raw_config)

    app_cfg = load_config(config_path)
    if app_cfg.seed is not None:
        _set_seed(app_cfg.seed)

    rng = random.Random(app_cfg.seed) if app_cfg.seed is not None else None

    env = make_game(app_cfg.game.id, **app_cfg.game.params)
    legal_mask = getattr(env, "legal_actions_mask", None)
    if legal_mask is None:
        raise ValueError("Environment must expose legal_actions_mask.")
    num_actions = int(len(legal_mask))

    agent_params: Dict[str, Any] = dict(app_cfg.agent.params)
    agent_params.setdefault("num_actions", num_actions)
    agent_params.setdefault("action_space", DiscreteActionSpace(num_actions))
    obs_builder = getattr(env, "observation_builder", None)
    if obs_builder is not None:
        agent_params.setdefault("observation_shape", obs_builder.observation_shape)
        agent_params.setdefault("observation_type", getattr(obs_builder, "observation_type", "board"))

    device = _resolve_device(agent_params.get("device"))
    agent_params["device"] = device

    agent = make_agent(app_cfg.agent.id, **agent_params)
    if hasattr(agent, "train"):
        agent.train()

    meta = collect_meta(
        config_path=config_path,
        command=command or " ".join(sys.argv),
        seed=app_cfg.seed,
        device=device,
    )
    write_meta_json(run_dir / "meta.json", meta)

    # Build callbacks
    callbacks: List[TrainerCallback] = [
        StatusFileCallback(
            run_dir=run_dir,
            interval=status_interval,
            total_steps=app_cfg.train.total_steps,
        ),
        _build_checkpoint_callback(run_dir, app_cfg.train.callbacks),
    ]
    for cb_cfg in app_cfg.train.callbacks:
        built = _build_metrics_file_callback(run_dir, cb_cfg)
        if built is not None:
            callbacks.append(built)
            continue
        for builder in (_build_epsilon_decay_callback, _build_lr_decay_callback):
            built = builder(cb_cfg)
            if built is not None:
                callbacks.append(built)
                break

    opponent_sampler = _build_opponent_sampler(app_cfg, agent, device)
    start_policy = _resolve_start_policy(app_cfg.train.start_policy)
    ro = getattr(app_cfg.train, "random_opening", None)
    random_opening_config = RandomOpeningConfig(**ro) if ro else None

    trainer = Trainer(
        env,
        agent,
        callbacks=callbacks,
        track_timings=app_cfg.train.track_timings,
        opponent_sampler=opponent_sampler,
        start_policy=start_policy,
        rng=rng,
        random_opening_config=random_opening_config,
    )

    # Install signal handlers for graceful stop
    _current_trainer = trainer
    _install_signal_handlers()

    try:
        trainer.train(
            total_steps=app_cfg.train.total_steps,
            deterministic=app_cfg.train.deterministic,
        )
    finally:
        _current_trainer = None
        _restore_signal_handlers()
        _remove_pid(run_dir)
