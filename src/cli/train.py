"""Hydra-powered training entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.config import AppConfig, CallbackConfig
from src.features.action_space import DiscreteActionSpace
from src.registry import make_agent, make_game
from src.training.trainer import (
    CheckpointCallback,
    EarlyStopCallback,
    EvalCallback,
    Trainer,
    TrainerCallback,
    WandbLoggerCallback,
    CSVLoggerCallback,
)


CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "configs")


def _make_eval_fn(app_cfg: AppConfig, eval_params: Dict[str, Any]):
    num_episodes = int(eval_params.get("num_episodes", app_cfg.eval.num_episodes))
    deterministic = bool(eval_params.get("deterministic", app_cfg.eval.deterministic))

    def eval_fn(trainer: Trainer) -> Dict[str, float]:
        agent = trainer.agent
        eval_env = make_game(app_cfg.game.id, **app_cfg.game.params)

        # Preserve agent mode
        restore = getattr(agent, "training", None)
        if hasattr(agent, "eval"):
            agent.eval()

        total_reward = 0.0
        total_length = 0

        for _ in range(num_episodes):
            obs = eval_env.reset()
            done = False
            while not done:
                legal_mask = getattr(eval_env, "legal_actions_mask", None)
                action = agent.act(obs, legal_mask=legal_mask, deterministic=deterministic)
                step = eval_env.step(action)
                total_reward += step.reward
                total_length += 1
                obs = step.obs
                done = step.done

        if restore is not None and restore and hasattr(agent, "train"):
            agent.train()

        return {
            "eval/avg_reward": total_reward / num_episodes,
            "eval/avg_length": total_length / num_episodes,
        }

    return eval_fn


def _build_callbacks(app_cfg: AppConfig, callback_cfgs: List[CallbackConfig]) -> List[TrainerCallback]:
    callbacks: List[TrainerCallback] = []
    run_dir = Path(HydraConfig.get().runtime.output_dir)

    for cb_cfg in callback_cfgs:
        if not cb_cfg.enabled:
            continue
        params = dict(cb_cfg.params)
        cb_type = cb_cfg.type.lower()

        if cb_type == "eval":
            interval = int(params.get("interval", 1000))
            eval_fn = _make_eval_fn(app_cfg, params or {})
            callbacks.append(EvalCallback(eval_fn, interval=interval, name=params.get("name", "eval")))
        elif cb_type == "checkpoint":
            output_dir = Path(params.get("output_dir", run_dir / "checkpoints"))
            interval = int(params.get("interval", 1000))
            prefix = params.get("prefix", "checkpoint")
            callbacks.append(CheckpointCallback(output_dir, interval=interval, prefix=prefix))
        elif cb_type == "csv":
            csv_path = Path(params.get("path", run_dir / "metrics.csv"))
            fieldnames = params.get("fieldnames")
            callbacks.append(CSVLoggerCallback(csv_path, fieldnames=fieldnames))
        elif cb_type == "wandb":
            callbacks.append(WandbLoggerCallback(**params))
        elif cb_type == "early_stop":
            monitor = params.get("monitor", "eval/avg_reward")
            patience = int(params.get("patience", 5))
            mode = params.get("mode", "max")
            callbacks.append(EarlyStopCallback(monitor=monitor, patience=patience, mode=mode))
        else:
            raise ValueError(f"Unknown callback type: {cb_cfg.type}")

    return callbacks


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    app_cfg = AppConfig.from_dict(cfg_dict)  # type: ignore[arg-type]

    env = make_game(app_cfg.game.id, **app_cfg.game.params)

    agent_params: Dict[str, Any] = dict(app_cfg.agent.params)
    legal_mask = getattr(env, "legal_actions_mask", None)
    if legal_mask is None:
        raise ValueError("Environment must expose legal_actions_mask for trainer configuration.")
    num_actions = int(len(legal_mask))
    agent_params.setdefault("num_actions", num_actions)
    agent_params.setdefault("action_space", DiscreteActionSpace(num_actions))

    observation_builder = getattr(env, "observation_builder", None)
    if observation_builder is not None:
        agent_params.setdefault("observation_shape", observation_builder.observation_shape)
        agent_params.setdefault("observation_type", getattr(observation_builder, "observation_type", "board"))

    agent = make_agent(app_cfg.agent.id, **agent_params)
    if hasattr(agent, "train"):
        agent.train()

    callbacks = _build_callbacks(app_cfg, app_cfg.train.callbacks)

    trainer = Trainer(env, agent, callbacks=callbacks, track_timings=app_cfg.train.track_timings)
    trainer.train(total_steps=app_cfg.train.total_steps, deterministic=app_cfg.train.deterministic)


if __name__ == "__main__":
    main()

