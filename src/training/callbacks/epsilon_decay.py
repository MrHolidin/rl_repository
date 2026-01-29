"""Epsilon decay callback: decay exploration rate each step or episode."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from src.training.trainer import TrainerCallback, Transition

if TYPE_CHECKING:
    from src.training.trainer import Trainer


class EpsilonDecayCallback(TrainerCallback):
    """Decay epsilon after each step or episode."""

    def __init__(self, every: str = "step"):
        if every not in {"step", "episode"}:
            raise ValueError("every must be 'step' or 'episode'")
        self.every = every

    def on_step_end(
        self,
        trainer: "Trainer",
        step: int,
        transition: Transition,
        metrics: dict,
    ) -> None:
        if self.every == "step":
            self._decay(trainer)

    def on_episode_end(
        self,
        trainer: "Trainer",
        episode: int,
        episode_info: Dict[str, Any],
    ) -> None:
        if self.every == "episode":
            self._decay(trainer)

    @staticmethod
    def _decay(trainer: "Trainer") -> None:
        agent = trainer.agent
        if not getattr(agent, "training", True):
            return
        if not (
            hasattr(agent, "epsilon")
            and hasattr(agent, "epsilon_decay")
            and hasattr(agent, "epsilon_min")
        ):
            return
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            if agent.epsilon < agent.epsilon_min:
                agent.epsilon = agent.epsilon_min
