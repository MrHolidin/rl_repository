"""Configuration schemas for Hydra composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


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
class TrainConfig:
    total_steps: int = 10000
    deterministic: bool = False
    track_timings: bool = False
    callbacks: List[CallbackConfig] = field(default_factory=list)


@dataclass
class EvalConfig:
    enabled: bool = False
    num_episodes: int = 10
    deterministic: bool = True


@dataclass
class AppConfig:
    game: GameConfig
    agent: AgentConfig
    train: TrainConfig
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        game = GameConfig(**data["game"])
        agent = AgentConfig(**data["agent"])

        train_data = data.get("train", {})
        callbacks = [CallbackConfig(**cb) for cb in train_data.get("callbacks", [])]
        train = TrainConfig(
            total_steps=train_data.get("total_steps", 10000),
            deterministic=train_data.get("deterministic", False),
            track_timings=train_data.get("track_timings", False),
            callbacks=callbacks,
        )

        eval_cfg = EvalConfig(**data.get("eval", {}))

        return cls(game=game, agent=agent, train=train, eval=eval_cfg)

