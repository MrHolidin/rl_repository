"""Configuration schema for training runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class GameConfig:
    id: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentConfig:
    id: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CallbackConfig:
    type: str
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OpponentSamplerConfig:
    type: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    opponent_sampler: OpponentSamplerConfig  # required
    total_steps: int = 10000
    # Stop after this many completed episodes (whichever hits first: this or total_steps).
    max_episodes: Optional[int] = None
    deterministic: bool = False
    track_timings: bool = False
    callbacks: List[CallbackConfig] = field(default_factory=list)
    start_policy: str = "random"
    random_opening: Optional[Dict[str, Any]] = None
    apply_augmentation: bool = False


@dataclass
class EvalConfig:
    enabled: bool = False
    num_episodes: int = 10
    deterministic: bool = True


@dataclass
class DistributedConfig:
    workers: int = 4
    worker_device: str = "cpu"
    checkpoint: Optional[str] = None  # path to starting .pt; None → fresh start
    # Directory of periodic checkpoints (or explicit list) to seed the frozen
    # self-play pool on restart. Each checkpoint = one past-self opponent; the
    # most recent ``max_frozen_agents`` survive eviction. None → empty pool.
    restore_frozen_from: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DistributedConfig":
        return cls(
            workers=int(data.get("workers", 4)),
            worker_device=str(data.get("worker_device", "cpu")),
            checkpoint=str(data["checkpoint"]) if data.get("checkpoint") else None,
            restore_frozen_from=(
                str(data["restore_frozen_from"])
                if data.get("restore_frozen_from")
                else None
            ),
        )


@dataclass
class AppConfig:
    game: GameConfig
    agent: AgentConfig
    train: TrainConfig
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: Optional[int] = None
    distributed: Optional[DistributedConfig] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        game = GameConfig(**data["game"])
        agent = AgentConfig(**data["agent"])

        train_data = data.get("train", {})
        callbacks = [
            CallbackConfig(
                type=cb["type"],
                enabled=cb.get("enabled", True),
                params=cb.get("params", {}),
            )
            for cb in train_data.get("callbacks", [])
        ]
        os_cfg = train_data.get("opponent_sampler")
        if os_cfg is None:
            raise ValueError("train.opponent_sampler is required")
        opponent_sampler = OpponentSamplerConfig(
            type=os_cfg.get("type", "random"),
            params=os_cfg.get("params", {}),
        )

        me = train_data.get("max_episodes")
        max_episodes = int(me) if me is not None else None

        train = TrainConfig(
            opponent_sampler=opponent_sampler,
            total_steps=train_data.get("total_steps", 10000),
            max_episodes=max_episodes,
            deterministic=train_data.get("deterministic", False),
            track_timings=train_data.get("track_timings", False),
            callbacks=callbacks,
            start_policy=str(train_data.get("start_policy", "random")),
            random_opening=train_data.get("random_opening"),
            apply_augmentation=bool(train_data.get("apply_augmentation", False)),
        )

        eval_data = data.get("eval", {})
        eval_cfg = EvalConfig(
            enabled=eval_data.get("enabled", False),
            num_episodes=eval_data.get("num_episodes", 10),
            deterministic=eval_data.get("deterministic", True),
        )

        seed = data.get("seed")
        if seed is not None:
            seed = int(seed)

        dist_data = data.get("distributed")
        distributed = DistributedConfig.from_dict(dist_data) if dist_data else None

        return cls(game=game, agent=agent, train=train, eval=eval_cfg, seed=seed, distributed=distributed)


def load_config(path: Union[str, Path]) -> AppConfig:
    """Load AppConfig from a YAML file."""
    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(data)}")
    return AppConfig.from_dict(data)
