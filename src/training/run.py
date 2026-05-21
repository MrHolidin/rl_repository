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
from src.training.opponent_sampler import (
    OpponentSampler,
    RandomOpponentSampler,
)
from src.training.callbacks import (
    CheckpointCallback,
    EpsilonDecayCallback,
    LearningRateDecayCallback,
    MetricsFileCallback,
    StatusFileCallback,
    TrainerCallback,
)
from src.training.random_opening import RandomOpeningConfig
from src.training.agent_perspective_env import (
    AgentPerspectiveEnv,
    make_minibg_shaping_fn,
)
from src.training.bglike_perspective import (
    make_bglike_agent_perspective_env,
    make_bglike_shaping_fn,
)
from src.training.trainer import StartPolicy, Trainer
from src.training.connect4_augmentations import make_connect4_horizontal_augmenter
from src.training.othello_augmentations import make_othello_d4_augmenter

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
    *,
    agent_id: str,
) -> Optional[TrainerCallback]:
    if not cb.enabled or cb.type.lower() != "metrics_file":
        return None
    p = cb.params
    interval = int(p.get("interval", 100))
    filename = str(p.get("filename", "metrics.csv"))
    preset = str(p.get("preset", "auto")).strip().lower()
    raw_columns = p.get("columns")

    columns: Optional[list[str]] = None
    if raw_columns is not None:
        if not isinstance(raw_columns, (list, tuple)):
            raise TypeError(
                "metrics_file callback 'columns' must be a list of column names "
                f"(got {type(raw_columns).__name__})"
            )
        columns = [str(x).strip() for x in raw_columns if str(x).strip()]

    from src.training.metrics_presets import resolve_metrics_csv_fieldnames

    fieldnames = resolve_metrics_csv_fieldnames(agent_id, preset=preset, columns=columns)
    return MetricsFileCallback(
        run_dir=run_dir,
        interval=interval,
        filename=filename,
        fieldnames=fieldnames,
    )


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


def _start_policy_to_probability(policy: StartPolicy) -> float:
    if policy == StartPolicy.AGENT_FIRST:
        return 1.0
    if policy == StartPolicy.OPPONENT_FIRST:
        return 0.0
    return 0.5


def _build_scripted_spec(game_id: str, p: Dict[str, Any]) -> "ScriptedOpponentsSpec":
    """Unified scripted opponents: ``scripted.distribution``; legacy keys still accepted."""
    from src.envs.minibg.heuristic_bots.bots import default_bot_constructors
    from src.training.selfplay.opponent_pool import ScriptedOpponentsSpec

    g = (game_id or "").strip().lower()
    scripted_block = dict(p.get("scripted") or {})
    dist_in: Dict[str, Any] = dict(scripted_block.get("distribution") or {})
    valid_bg = frozenset(default_bot_constructors().keys())

    def _as_weights(d: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, v in d.items():
            key = str(k).strip()
            if not key:
                continue
            out[key] = max(0.0, float(v))
        return out

    if g in ("minibg", "bglike"):
        if g == "minibg":
            valid_src = default_bot_constructors
        else:
            from src.envs.bglike.heuristic_bots import default_bot_constructors as valid_src

        valid_bg = frozenset(valid_src().keys())
        if dist_in:
            w = _as_weights(dist_in)
            for name in w:
                if name != "random" and name not in valid_bg:
                    raise ValueError(
                        f"{g} scripted.distribution: unknown key {name!r}; "
                        f"use 'random' or {sorted(valid_bg)}"
                    )
            return ScriptedOpponentsSpec(g, w)

        raw_bots = p.get("minibg_bots") or p.get("bots")
        bots = [str(x).strip() for x in (raw_bots or []) if str(x).strip()]
        if not bots:
            raise ValueError(
                f"game.id {g}: set opponent_sampler.params.scripted.distribution, "
                "or legacy minibg_bots / bots (non-empty)"
            )
        for b in bots:
            if b not in valid_bg:
                raise ValueError(f"{g} unknown bot {b!r}; valid: {sorted(valid_bg)}")
        if bool(p.get("equal_opponent_mass", False)):
            share = 1.0 / (1 + len(bots))
            dweights = {"random": share}
            per = (1.0 - share) / len(bots)
            for b in bots:
                dweights[b] = per
            return ScriptedOpponentsSpec(g, dweights)
        rf = p.get("minibg_random_fraction", p.get("random_fraction"))
        if rf is None:
            rf = 0.5
        rf = min(1.0, max(0.0, float(rf)))
        dweights = {"random": rf}
        if bots:
            per = (1.0 - rf) / len(bots)
            for b in bots:
                dweights[b] = per
        return ScriptedOpponentsSpec(g, dweights)

    if dist_in:
        return ScriptedOpponentsSpec("classic", _as_weights(dist_in))
    h = dict(p.get("heuristic_distribution") or {})
    if h:
        return ScriptedOpponentsSpec("classic", _as_weights(h))
    return ScriptedOpponentsSpec(
        "classic",
        {
            "random": 0.2,
            "heuristic": 0.5,
            "smart_heuristic": 0.3,
        },
    )


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
        p = dict(cfg.params or {})
        use_seed = p.get("seed")
        if use_seed is None:
            use_seed = seed
        if use_seed is None:
            raise ValueError(
                "opponent_sampler type 'pool' requires app seed or opponent_sampler.params.seed"
            )
        use_seed = int(use_seed)
        game_id = (app_cfg.game.id or "").strip().lower()
        sp_raw = p.get("self_play")
        if not sp_raw:
            if game_id in ("minibg", "bglike"):
                self_play_config = None
            else:
                sp = {}
                save_every = int(sp.get("save_every", 1000))
                self_play_config = SelfPlayConfig(
                    start_episode=int(sp.get("start_episode", 0)),
                    current_self_fraction=float(sp.get("current_self_fraction", 0.3)),
                    past_self_fraction=float(sp.get("past_self_fraction", 0.3)),
                    max_frozen_agents=int(sp.get("max_frozen_agents", 10)),
                    save_every=max(1, save_every),
                    frozen_ema_beta=float(sp.get("frozen_ema_beta", 0.05)),
                )
        else:
            sp = dict(sp_raw)
            save_every = int(sp.get("save_every", 1000))
            self_play_config = SelfPlayConfig(
                start_episode=int(sp.get("start_episode", 0)),
                current_self_fraction=float(sp.get("current_self_fraction", 0.3)),
                past_self_fraction=float(sp.get("past_self_fraction", 0.3)),
                max_frozen_agents=int(sp.get("max_frozen_agents", 10)),
                save_every=max(1, save_every),
                frozen_ema_beta=float(sp.get("frozen_ema_beta", 0.05)),
            )
        scripted = _build_scripted_spec(game_id, p)
        pool = OpponentPool(
            device=device,
            seed=use_seed,
            self_play_config=self_play_config,
            scripted=scripted,
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

    game_params = dict(app_cfg.game.params or {})
    # Legacy/no-op param: control flow is handled by AgentPerspectiveEnv now.
    game_params.pop("control_driver", None)
    game_id = (app_cfg.game.id or "").strip().lower()

    agent_params: Dict[str, Any] = dict(app_cfg.agent.params)
    base_env = None
    if game_id == "bglike":
        from src.envs.bglike.action_map import NUM_ENV_ACTIONS
        from src.envs.bglike.obs import OBS_DIM

        num_actions = int(NUM_ENV_ACTIONS)
        agent_params.setdefault("num_actions", num_actions)
        agent_params.setdefault("action_space", DiscreteActionSpace(num_actions))
        agent_params.setdefault("observation_shape", (OBS_DIM,))
        agent_params.setdefault("observation_type", "vector")
    else:
        base_env = make_game(app_cfg.game.id, **game_params)
        legal_mask = getattr(base_env, "legal_actions_mask", None)
        if legal_mask is None:
            raise ValueError("Environment must expose legal_actions_mask.")
        num_actions = int(len(legal_mask))
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
        built = _build_metrics_file_callback(run_dir, cb_cfg, agent_id=app_cfg.agent.id)
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

    data_augment_fn: Optional[Callable] = None
    if getattr(app_cfg.train, "apply_augmentation", False):
        if game_id == "connect4":
            num_cols = getattr(base_env, "cols", 7)
            data_augment_fn = make_connect4_horizontal_augmenter(num_cols)
        elif game_id == "othello":
            board_size = getattr(base_env, "size", 8)
            data_augment_fn = make_othello_d4_augmenter(board_size)
        else:
            raise ValueError(
                f"apply_augmentation is True but no augmentations defined for game '{app_cfg.game.id}'"
            )

    shaping_fn = None
    if game_id == "minibg":
        shaping_fn = make_minibg_shaping_fn(
            float(game_params.get("battle_damage_shaping", 0.0))
        )
    elif game_id == "bglike":
        shaping_fn = make_bglike_shaping_fn(
            float(game_params.get("battle_damage_shaping", 0.0))
        )

    if game_id == "bglike":
        lobby_kw = {
            k: v
            for k, v in game_params.items()
            if k not in ("num_current_seats", "battle_damage_shaping", "seed")
        }
        env_seed = app_cfg.seed if app_cfg.seed is not None else game_params.get("seed")
        env = make_bglike_agent_perspective_env(
            opponent_sampler,
            num_current_seats=int(game_params.get("num_current_seats", 1)),
            seed=env_seed,
            shaping_fn=shaping_fn,
            rng=rng,
            **lobby_kw,
        )
        env.set_learner_agent(agent)
    else:
        env = AgentPerspectiveEnv(
            base_env=base_env,
            opponent_sampler=opponent_sampler,
            agent_first_probability=_start_policy_to_probability(start_policy),
            rng=rng,
            random_opening_config=random_opening_config,
            shaping_fn=shaping_fn,
        )

    trainer = Trainer(
        env,
        agent,
        callbacks=callbacks,
        track_timings=app_cfg.train.track_timings,
        opponent_sampler=opponent_sampler,
        data_augment_fn=data_augment_fn,
        max_episodes=app_cfg.train.max_episodes,
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
