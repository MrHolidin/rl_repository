"""Distributed PPO training pipeline — mirrors run.py, adds distributed: config block."""

from __future__ import annotations

import signal
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Union

from src.agents.ppo_structured_minibg_agent import MiniBGPPOStructuredAgent
from src.config import load_config
from src.features.action_space import DiscreteActionSpace
from src.models.ppo_policy_factory import (
    PPO_NETWORK_BGLIKE_STRUCTURED,
    PPO_NETWORK_BGLIKE_STRUCTURED_V2,
    PPO_NETWORK_BGLIKE_STRUCTURED_V3,
    PPO_NETWORK_BGLIKE_STRUCTURED_V4,
    PPO_NETWORK_BGLIKE_STRUCTURED_V5,
    PPO_NETWORK_BGLIKE_STRUCTURED_V6,
    PPO_NETWORK_BGLIKE_STRUCTURED_V7,
    PPO_NETWORK_BGLIKE_STRUCTURED_V8,
    PPO_NETWORK_BGLIKE_STRUCTURED_V9,
    PPO_NETWORK_BGLIKE_STRUCTURED_V10,
    PPO_NETWORK_BGLIKE_STRUCTURED_V11,
    PPO_NETWORK_BGLIKE_STRUCTURED_V11_HEROES,
    PPO_NETWORK_MINIBG_STRUCTURED,
)
from src.training.bg_network_policy import (
    reject_flat_bg_network,
    validate_heroes_consistency,
)
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
from src.training.selfplay.league_config import parse_league_settings

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

    # --- Agent / env sizing (mirrors run.py; workers read game_id from game_params) ---
    game_id = (app_cfg.game.id or "").strip().lower()
    game_params = dict(app_cfg.game.params or {})
    game_params.pop("control_driver", None)
    game_params["game_id"] = game_id

    agent_params = dict(app_cfg.agent.params)
    network_type = str(agent_params.get("network_type", "")).strip().lower()
    reject_flat_bg_network(game_id, network_type)
    use_structured = network_type in (
        PPO_NETWORK_MINIBG_STRUCTURED,
        PPO_NETWORK_BGLIKE_STRUCTURED,
        PPO_NETWORK_BGLIKE_STRUCTURED_V2,
        PPO_NETWORK_BGLIKE_STRUCTURED_V3,
        PPO_NETWORK_BGLIKE_STRUCTURED_V4,
        PPO_NETWORK_BGLIKE_STRUCTURED_V5,
        PPO_NETWORK_BGLIKE_STRUCTURED_V6,
        PPO_NETWORK_BGLIKE_STRUCTURED_V7,
        PPO_NETWORK_BGLIKE_STRUCTURED_V8,
        PPO_NETWORK_BGLIKE_STRUCTURED_V9,
        PPO_NETWORK_BGLIKE_STRUCTURED_V10,
        PPO_NETWORK_BGLIKE_STRUCTURED_V11,
        PPO_NETWORK_BGLIKE_STRUCTURED_V11_HEROES,
    )
    game_params["use_structured"] = use_structured
    # DvD/v7: thread the population knobs through to workers (mg) so each worker
    # builds a PPODvDAgent with the right identity count / repulsion strength.
    is_dvd_v7 = network_type in (
        PPO_NETWORK_BGLIKE_STRUCTURED_V7,
        PPO_NETWORK_BGLIKE_STRUCTURED_V8,
        PPO_NETWORK_BGLIKE_STRUCTURED_V9,
        PPO_NETWORK_BGLIKE_STRUCTURED_V10,
        PPO_NETWORK_BGLIKE_STRUCTURED_V11,
        PPO_NETWORK_BGLIKE_STRUCTURED_V11_HEROES,
    )
    if is_dvd_v7:
        game_params["dvd_network_type"] = network_type
        game_params["dvd_num_identities"] = int(agent_params.get("num_identities", 8))
        game_params["dvd_diversity_coef"] = float(agent_params.get("diversity_coef", 0.0))
        game_params["dvd_diversity_ema"] = float(agent_params.get("diversity_ema", 0.1))
        # Per-identity tribe assignment MUST reach the worker that computes the
        # bonus, else _resolve_identity_tribes(None) auto-assigns a *different*
        # tribe set ([None, BEAST, DEMON, MECH...]) and every identity is
        # rewarded for the wrong tribe (the dist_ppo_078 collapse bug).
        game_params["dvd_identity_tribes"] = agent_params.get("identity_tribes")
        game_params["dvd_reward_mode"] = str(agent_params.get("diversity_reward_mode", "final"))
        game_params["dvd_sibling_fraction"] = float(
            agent_params.get("sibling_fraction", 0.5)
        )
    # v5 / v6 require the per-ability ``obs_kind="bglike_v5"`` layout. We
    # auto-pin it here so existing v3-shaped configs don't accidentally feed
    # them the smaller obs (which would fail the model's obs_dim check at
    # startup).
    if (
        network_type
        in (
            PPO_NETWORK_BGLIKE_STRUCTURED_V5,
            PPO_NETWORK_BGLIKE_STRUCTURED_V6,
            PPO_NETWORK_BGLIKE_STRUCTURED_V7,
            PPO_NETWORK_BGLIKE_STRUCTURED_V8,
            PPO_NETWORK_BGLIKE_STRUCTURED_V9,
            PPO_NETWORK_BGLIKE_STRUCTURED_V10,
            PPO_NETWORK_BGLIKE_STRUCTURED_V11,
        )
        and game_id == "bglike"
    ):
        from src.envs.bglike.lobby_env import OBS_KIND_BGLIKE_V5

        existing = game_params.get("obs_kind")
        if existing is not None and existing != OBS_KIND_BGLIKE_V5:
            raise ValueError(
                f"network_type={network_type!r} requires obs_kind={OBS_KIND_BGLIKE_V5!r}, "
                f"got {existing!r}"
            )
        game_params["obs_kind"] = OBS_KIND_BGLIKE_V5

    # v11_heroes reads the hero-augmented obs (obs_v5 + hero block).
    if network_type == PPO_NETWORK_BGLIKE_STRUCTURED_V11_HEROES and game_id == "bglike":
        from src.envs.bglike.lobby_env import OBS_KIND_BGLIKE_V5_HEROES

        existing = game_params.get("obs_kind")
        if existing is not None and existing != OBS_KIND_BGLIKE_V5_HEROES:
            raise ValueError(
                f"network_type={network_type!r} requires obs_kind={OBS_KIND_BGLIKE_V5_HEROES!r}, "
                f"got {existing!r}"
            )
        game_params["obs_kind"] = OBS_KIND_BGLIKE_V5_HEROES
        # The hero net is meaningless without heroes assigned; default it on
        # (an explicit game.params.with_heroes still wins).
        game_params.setdefault("with_heroes", True)

    # Reject a heroes/network mismatch (e.g. with_heroes=true on a non-hero net).
    validate_heroes_consistency(game_id, network_type, game_params)

    if game_id in ("minibg", "bglike"):
        from src.training.patch_config import apply_patch_to_agent_params

        apply_patch_to_agent_params(game_params, agent_params)

    if game_id == "bglike":
        from src.envs.bglike.action_map import NUM_ENV_ACTIONS
        from src.training.obs_sizing import apply_bg_observation_defaults

        num_actions = int(NUM_ENV_ACTIONS)
        apply_bg_observation_defaults(
            game_id, agent_params, obs_kind=game_params.get("obs_kind")
        )
    elif game_id == "minibg":
        from src.envs.minibg.action_map import NUM_ENV_ACTIONS
        from src.training.obs_sizing import apply_bg_observation_defaults

        num_actions = int(NUM_ENV_ACTIONS)
        apply_bg_observation_defaults(game_id, agent_params)
    else:
        base_env = make_game(app_cfg.game.id, **game_params)
        legal_mask = getattr(base_env, "legal_actions_mask", None)
        if legal_mask is None:
            raise ValueError("Environment must expose legal_actions_mask.")
        num_actions = int(len(legal_mask))
        obs_builder = getattr(base_env, "observation_builder", None)
        if obs_builder is not None:
            agent_params.setdefault("observation_shape", obs_builder.observation_shape)
            agent_params.setdefault(
                "observation_type", getattr(obs_builder, "observation_type", "board")
            )

    agent_params.setdefault("num_actions", num_actions)
    agent_params.setdefault("action_space", DiscreteActionSpace(num_actions))
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
        if is_dvd_v7:
            from src.agents.ppo_dvd_agent import PPODvDAgent

            agent = PPODvDAgent.load(
                str(ck_path),
                device=device,
                seed=seed,
                patch_build=agent_params.get("patch_build"),
                diversity_coef=float(agent_params.get("diversity_coef", 0.0)),
                diversity_ema=float(agent_params.get("diversity_ema", 0.1)),
                # Pass identity_tribes so the host's dvd_* metrics resolve the
                # configured tribes (workers already get these via game_params;
                # without this the host auto-assigns the wrong tribe set on resume
                # and dvd_assigned_frac_i is mislabelled). Do NOT pass
                # identity_init_std → the trained gate is kept, not re-randomised.
                identity_tribes=agent_params.get("identity_tribes"),
                diversity_reward_mode=str(agent_params.get("diversity_reward_mode", "final")),
            )
        elif use_structured:
            agent = MiniBGPPOStructuredAgent.load(
                str(ck_path),
                device=device,
                seed=seed,
                patch_build=agent_params.get("patch_build"),
                update_opt_mode=agent_params.get("update_opt_mode", "compile"),
            )
        else:
            from src.agents.ppo_agent import PPOAgent

            agent = PPOAgent.load(str(ck_path), device=device, seed=seed)
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
    opp_params = dict(app_cfg.train.opponent_sampler.params or {})
    league_settings = parse_league_settings(opp_params)
    sp = dict(opp_params.get("self_play") or {})
    current_fraction = league_settings.sampler.current_self_fraction
    past_fraction = league_settings.sampler.past_self_fraction
    start_episode = int(sp.get("start_episode", 0))
    max_pool_size = int(sp.get("max_frozen_agents", 20))
    ema_beta = league_settings.rating.ema_beta
    rating = league_settings.rating.kind
    trueskill_cfg = league_settings.rating.trueskill
    sampler_kind = league_settings.sampler.kind

    scripted_spec = _build_scripted_spec(app_cfg.game.id, opp_params)
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

    # Optionally re-seed the frozen self-play pool from a run's checkpoints dir
    # (each periodic checkpoint == one historical freeze). Keep only files with a
    # numeric step suffix (skip init.pt / *_final.pt); the trainer caps to
    # max_frozen_agents, keeping the most-recent past selves.
    frozen_pool_checkpoints: list = []
    if dist_cfg.restore_frozen_from:

        def _ckpt_step(path: Path) -> Optional[int]:
            stem = path.stem  # e.g. dist_bglike_ppo_v6_74257_4012425
            tail = stem.rsplit("_", 1)[-1]
            return int(tail) if tail.isdigit() else None

        src = Path(dist_cfg.restore_frozen_from)
        if src.is_dir():
            cands = sorted(
                (p for p in src.glob("*.pt") if _ckpt_step(p) is not None),
                key=lambda p: _ckpt_step(p),
            )
        elif src.is_file():
            cands = [src]
        else:
            cands = []
            print(f"[dist_ppo] restore_frozen_from not found: {src}", flush=True)
        frozen_pool_checkpoints = [str(p) for p in cands]
        print(
            f"[dist_ppo] restore_frozen_from={src}: {len(frozen_pool_checkpoints)} checkpoint(s)",
            flush=True,
        )

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
        rating=rating,
        trueskill=trueskill_cfg,
        sampler_kind=sampler_kind,
        start_episode=start_episode,
        run_dir=str(run_dir),
        frozen_pool_checkpoints=frozen_pool_checkpoints,
    )

    _current_trainer = trainer
    _install_signal_handlers()

    try:
        trainer.train(total_steps)
    finally:
        _current_trainer = None
        _restore_signal_handlers()
        _remove_pid(run_dir)
