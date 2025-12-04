"""Generic training loop and callbacks for turn-based environments."""

from __future__ import annotations
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.agents.base_agent import BaseAgent
from src.envs.base import StepResult, TurnBasedEnv
from src.training.opponent_sampler import OpponentSampler, RandomOpponentSampler
from src.training.random_opening import RandomOpeningConfig


@dataclass
class Transition:
    """Container representing a single environment interaction."""

    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    terminated: bool
    truncated: bool
    info: Dict[str, Any]
    legal_mask: Optional[np.ndarray] = None
    next_legal_mask: Optional[np.ndarray] = None

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


@dataclass
class PendingTransition:
    """Partial transition awaiting opponent response."""

    obs: np.ndarray
    action: int
    reward: float
    legal_mask: Optional[np.ndarray]


class StartPolicy(Enum):
    """Policy for determining who goes first in each episode."""

    RANDOM = "random"
    AGENT_FIRST = "agent_first"
    OPPONENT_FIRST = "opponent_first"


class TrainerCallback:
    """Base class for trainer callbacks."""

    def on_train_begin(self, trainer: "Trainer") -> None:
        pass

    def on_step_end(
        self,
        trainer: "Trainer",
        step: int,
        transition: Transition,
        metrics: Dict[str, float],
    ) -> None:
        pass

    def on_episode_end(
        self,
        trainer: "Trainer",
        episode: int,
        episode_info: Dict[str, Any],
    ) -> None:
        pass

    def on_train_end(self, trainer: "Trainer") -> None:
        pass


class EvalCallback(TrainerCallback):
    """Run evaluation at a fixed step interval."""

    def __init__(self, eval_fn: Callable[["Trainer"], Dict[str, float]], interval: int, name: str = "eval"):
        self.eval_fn = eval_fn
        self.interval = max(1, interval)
        self.name = name
        self.history: List[Dict[str, float]] = []
        self.time_spent: float = 0.0

    def on_step_end(
        self,
        trainer: "Trainer",
        step: int,
        transition: Transition,
        metrics: Dict[str, float],
    ) -> None:
        if step % self.interval == 0:
            start = perf_counter()
            result = self.eval_fn(trainer)
            self.time_spent += perf_counter() - start
            self.history.append(result)
            metrics.update({f"{self.name}/{k}": v for k, v in result.items()})
            summary = " | ".join(f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in result.items())
            print(f"[{self.name}] step {step}: {summary}")


class CheckpointCallback(TrainerCallback):
    """Persist agent checkpoints during training."""

    def __init__(self, output_dir: str | Path, interval: int, prefix: str = "checkpoint"):
        self.output_dir = Path(output_dir)
        self.interval = max(1, interval)
        self.prefix = prefix
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_step_end(
        self,
        trainer: "Trainer",
        step: int,
        transition: Transition,
        metrics: Dict[str, float],
    ) -> None:
        if step % self.interval == 0:
            path = self.output_dir / f"{self.prefix}_{step}.pt"
            trainer.agent.save(str(path))
            metrics["checkpoint_saved"] = step
            if trainer.opponent_sampler is not None:
                trainer.opponent_sampler.on_checkpoint(path, trainer.episode_index)


class WandbLoggerCallback(TrainerCallback):
    """Optional Weights & Biases logger."""

    def __init__(self, **wandb_init_kwargs: Any):
        try:
            import wandb  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("wandb is not installed. Install wandb or disable WandbLoggerCallback.") from exc
        self.wandb = wandb
        self.wandb_init_kwargs = wandb_init_kwargs
        self._run = None

    def on_train_begin(self, trainer: "Trainer") -> None:
        self._run = self.wandb.init(**self.wandb_init_kwargs)

    def on_step_end(
        self,
        trainer: "Trainer",
        step: int,
        transition: Transition,
        metrics: Dict[str, float],
    ) -> None:
        if self._run is not None and metrics:
            self.wandb.log(metrics, step=step)

    def on_train_end(self, trainer: "Trainer") -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None


class EpsilonDecayCallback(TrainerCallback):
    """Decay epsilon after each step or episode."""

    def __init__(self, every: str = "step"):
        """
        Args:
            every: "step" or "episode" - when to decay epsilon
        """
        if every not in {"step", "episode"}:
            raise ValueError("every must be 'step' or 'episode'")
        self.every = every

    def on_step_end(
        self,
        trainer: "Trainer",
        step: int,
        transition: Transition,
        metrics: Dict[str, float],
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
        """Decay epsilon if agent supports it."""
        agent = trainer.agent
        if not getattr(agent, "training", True):
            return
        if not (hasattr(agent, "epsilon") and hasattr(agent, "epsilon_decay") and hasattr(agent, "epsilon_min")):
            return
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            if agent.epsilon < agent.epsilon_min:
                agent.epsilon = agent.epsilon_min


class EarlyStopCallback(TrainerCallback):
    """Stop training when monitored metric satisfies condition."""

    def __init__(
        self,
        monitor: str,
        patience: int,
        mode: str = "max",
    ) -> None:
        self.monitor = monitor
        self.patience = patience
        if mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")
        self.mode = mode
        self._best: Optional[float] = None
        self._bad_steps = 0

    def on_step_end(
        self,
        trainer: "Trainer",
        step: int,
        transition: Transition,
        metrics: Dict[str, float],
    ) -> None:
        if self.monitor not in metrics:
            return
        value = metrics[self.monitor]
        improved = (
            self._best is None
            or (self.mode == "max" and value > self._best)
            or (self.mode == "min" and value < self._best)
        )
        if improved:
            self._best = value
            self._bad_steps = 0
        else:
            self._bad_steps += 1
            if self._bad_steps >= self.patience:
                trainer.stop_training = True


class LearningRateDecayCallback(TrainerCallback):
    """Multiply optimizer learning rate by ``decay_factor`` every N steps."""

    def __init__(
        self,
        *,
        interval_steps: int,
        decay_factor: float,
        min_lr: Optional[float] = None,
        optimizer_attr: str = "optimizer",
        metric_key: str = "learning_rate",
    ) -> None:
        if interval_steps <= 0:
            raise ValueError("interval_steps must be > 0")
        if decay_factor <= 0:
            raise ValueError("decay_factor must be > 0")
        self.interval_steps = interval_steps
        self.decay_factor = decay_factor
        self.min_lr = min_lr
        self.optimizer_attr = optimizer_attr
        self.metric_key = metric_key
        self._optimizer = None
        self._warned_missing_optimizer = False

    def on_train_begin(self, trainer: "Trainer") -> None:
        self._optimizer = getattr(trainer.agent, self.optimizer_attr, None)
        if self._optimizer is None and not self._warned_missing_optimizer:
            print(
                f"[LearningRateDecayCallback] Agent has no '{self.optimizer_attr}'. "
                "Skipping LR decay."
            )
            self._warned_missing_optimizer = True

    def on_step_end(
        self,
        trainer: "Trainer",
        step: int,
        transition: Transition,
        metrics: Dict[str, float],
    ) -> None:
        if self._optimizer is None:
            return
        if step % self.interval_steps != 0:
            return

        new_lr = None
        for group in self._optimizer.param_groups:
            old_lr = group.get("lr", None)
            if old_lr is None:
                continue
            updated = old_lr * self.decay_factor
            if self.min_lr is not None:
                updated = max(updated, self.min_lr)
            group["lr"] = updated
            new_lr = updated

        if new_lr is None:
            return

        agent = getattr(trainer, "agent", None)
        if agent is not None and hasattr(agent, "learning_rate"):
            agent.learning_rate = new_lr

        if self.metric_key:
            metrics[self.metric_key] = new_lr


class Trainer:
    """Generic training loop that interacts with an environment via an agent."""

    def __init__(
        self,
        env: TurnBasedEnv,
        agent: BaseAgent,
        callbacks: Optional[Iterable[TrainerCallback]] = None,
        track_timings: bool = False,
        opponent_sampler: Optional[OpponentSampler] = None,
        start_policy: StartPolicy = StartPolicy.RANDOM,
        rng: Optional[Union[random.Random, np.random.Generator]] = None,
        random_opening_config: Optional["RandomOpeningConfig"] = None,
    ) -> None:
        self.env = env
        self.agent = agent
        self.callbacks: List[TrainerCallback] = list(callbacks) if callbacks is not None else []
        self.global_step = 0
        self.episode_index = 0
        self.stop_training = False
        self.track_timings = track_timings
        self._timings: Dict[str, float] = {"total": 0.0, "core": 0.0, "callbacks": 0.0}
        self.timing_report: Optional[Dict[str, float]] = None
        self.opponent_sampler = opponent_sampler
        self._default_opponent_sampler = RandomOpponentSampler()
        self._current_opponent: Optional[BaseAgent] = None
        self._target_total_steps = 0
        self.start_policy = start_policy
        self._rng = rng if rng is not None else random
        self._action_dim: Optional[int] = None
        self._agent_token = 1  # +1 when agent moves first, -1 otherwise
        self.random_opening_config = random_opening_config

    def train(self, total_steps: int, *, deterministic: bool = False) -> None:
        """Run the training loop for the specified number of agent updates."""
        self._target_total_steps = total_steps
        reward_config = getattr(self.env, "reward_config", None)

        obs = self.env.reset()
        obs, prologue_done = self._apply_random_opening(obs)
        if prologue_done:
            obs = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        pending: Optional[PendingTransition] = None

        for callback in self.callbacks:
            callback.on_train_begin(self)

        self._prepare_opponent()

        # Determine who goes first in first episode
        current_player_is_agent = self._should_agent_go_first()

        # If opponent goes first, handle opening
        if not current_player_is_agent:
            obs, episode_reward, episode_length, current_player_is_agent = self._handle_opponent_start(
                obs, episode_reward, episode_length, reward_config, None
            )

        while self.global_step < total_steps and not self.stop_training:
            iteration_start = perf_counter() if self.track_timings else None

            if current_player_is_agent:
                # Agent's turn
                agent_step, pending = self._agent_turn(obs, deterministic)
                episode_length += 1

                if agent_step.done:
                    # Agent won or drew
                    transition = Transition(
                        obs=obs,
                        action=pending.action,
                        reward=agent_step.reward,
                        next_obs=agent_step.obs,
                        terminated=agent_step.terminated,
                        truncated=agent_step.truncated,
                        info=agent_step.info,
                        legal_mask=pending.legal_mask,
                        next_legal_mask=self._zero_mask_like(pending.legal_mask),
                    )
                    metrics = self._process_agent_transition(transition)
                    episode_reward += transition.reward
                    core_done = perf_counter() if self.track_timings else None
                    self._after_transition(transition, metrics, iteration_start, core_done)

                    obs, episode_reward, episode_length, pending, transition, end_info = self._handle_episode_end(
                        episode_reward, episode_length, agent_step.info
                    )
                    iteration_start = perf_counter() if self.track_timings else None
                    episode_reward = self._process_maybe_opening_transition(transition, episode_reward, iteration_start)
                    current_player_is_agent = self._should_agent_go_first()
                    continue

                # Game continues, wait for opponent
                obs = agent_step.obs
                current_player_is_agent = False
            else:
                # Opponent's turn
                if pending is None:
                    obs, episode_reward, episode_length, current_player_is_agent = self._handle_opponent_start(
                        obs, episode_reward, episode_length, reward_config, iteration_start
                    )
                    continue
                else:
                    # Normal opponent turn after agent's move
                    opponent_step, transition = self._opponent_turn_and_credit(obs, pending, reward_config)
                    episode_length += 1
                    metrics = self._process_agent_transition(transition)
                    episode_reward += transition.reward
                    pending = None

                    core_done = perf_counter() if self.track_timings else None
                    self._after_transition(transition, metrics, iteration_start, core_done)

                    obs = opponent_step.obs
                    if opponent_step.done:
                        obs, episode_reward, episode_length, pending, transition, end_info = self._handle_episode_end(
                            episode_reward, episode_length, opponent_step.info
                        )
                        iteration_start = perf_counter() if self.track_timings else None
                        episode_reward = self._process_maybe_opening_transition(transition, episode_reward, iteration_start)
                        current_player_is_agent = self._should_agent_go_first()
                        continue

                    current_player_is_agent = True  # Switch back to agent

        for callback in self.callbacks:
            callback.on_train_end(self)

        if self.track_timings:
            self._finalize_timings()

    def _process_agent_transition(self, transition: Transition) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        observe_metrics = self.agent.observe(transition) or {}
        metrics.update(observe_metrics)

        update_metrics = self.agent.update() or {}
        for key, value in update_metrics.items():
            if key in metrics and isinstance(metrics[key], (int, float)):
                metrics[key] = (metrics[key] + value) / 2.0
            else:
                metrics[key] = value
        return metrics

    def _after_transition(
        self,
        transition: Transition,
        metrics: Dict[str, float],
        iteration_start: Optional[float],
        core_done: Optional[float],
    ) -> None:
        if self.track_timings and iteration_start is not None and core_done is not None:
            self._timings["core"] += max(0.0, core_done - iteration_start)

        self.global_step += 1
        for callback in self.callbacks:
            cb_start = perf_counter() if self.track_timings else None
            callback.on_step_end(self, self.global_step, transition, metrics)
            if self.track_timings and cb_start is not None:
                self._timings["callbacks"] += perf_counter() - cb_start

        if self.track_timings and iteration_start is not None:
            self._timings["total"] += perf_counter() - iteration_start

    def _handle_episode_end(
        self,
        episode_reward: float,
        episode_length: int,
        info: Dict[str, Any],
    ) -> Tuple[np.ndarray, float, int, Optional[PendingTransition], Optional[Transition], Optional[Dict[str, Any]]]:
        """
        Handle episode end: callbacks, reset environment, prepare opponent.
        
        Returns:
            obs, episode_reward (reset to 0.0), episode_length (reset to 0), pending, transition, end_info
        """
        winner = info.get("winner") if isinstance(info, dict) else None
        agent_result = self._agent_relative_result(winner)
        episode_info = {
            "reward": episode_reward,
            "length": episode_length,
            "info": info,
            "agent_token": self._agent_token,
            "agent_result": agent_result,
        }
        for callback in self.callbacks:
            callback.on_episode_end(self, self.episode_index, episode_info)

        if self.opponent_sampler is not None:
            self.opponent_sampler.on_episode_end(self.episode_index, episode_info)

        obs = self.env.reset()
        self.episode_index += 1
        self._current_opponent = None

        if not self.stop_training and self.global_step < self._target_total_steps:
            self._prepare_opponent()

        obs, prologue_done = self._apply_random_opening(obs)
        if prologue_done:
            obs = self.env.reset()

        return obs, 0.0, 0, None, None, None

    def _prepare_opponent(self) -> BaseAgent:
        sampler = self.opponent_sampler or self._default_opponent_sampler
        sampler.prepare(self.episode_index)
        opponent = sampler.sample()
        opponent.eval()
        if hasattr(opponent, "epsilon"):
            setattr(opponent, "epsilon", 0.0)
        self._current_opponent = opponent
        return opponent

    def _as_bool_mask(self, mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Convert mask to boolean array."""
        if mask is None:
            return None
        return np.asarray(mask, dtype=bool, order='C')

    def _zero_mask_like(self, mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Create zero mask with same shape as input."""
        m = self._as_bool_mask(mask)
        return None if m is None else np.zeros_like(m, dtype=bool)

    def _get_action_dim(self) -> int:
        """Get the action space dimension from the environment."""
        if self._action_dim is not None:
            return self._action_dim
        # Try to get from legal_actions_mask
        if hasattr(self.env, "legal_actions_mask") and self.env.legal_actions_mask is not None:
            self._action_dim = int(len(self.env.legal_actions_mask))
            return self._action_dim
        # Try cols attribute (for Connect4Env)
        if hasattr(self.env, "cols"):
            self._action_dim = int(self.env.cols)
            return self._action_dim
        raise RuntimeError("Cannot infer action dimension; provide legal_actions_mask or cols.")

    def _opponent_opening(
        self,
        obs: np.ndarray,
        episode_length: int,
        reward_config: Any,
    ) -> Tuple[np.ndarray, int, Optional[Transition], Optional[Dict[str, Any]]]:
        """
        If opponent goes first in current episode, make their move.
        
        Returns:
            obs, episode_length, maybe_transition_or_None, maybe_episode_end_info_or_None
        """
        opponent = self._current_opponent or self._prepare_opponent()
        opp_mask = self._as_bool_mask(getattr(self.env, "legal_actions_mask", None))
        opp_action = opponent.act(obs, legal_mask=opp_mask, deterministic=False)
        opp_step = self.env.step(opp_action)
        episode_length += 1

        if not opp_step.done:
            return opp_step.obs, episode_length, None, None

        # Episode ended on opponent's first move: account for agent loss/draw
        winner = opp_step.info.get("winner")
        if reward_config is not None:
            final_reward = (
                getattr(reward_config, "draw", 0.0) if winner == 0
                else getattr(reward_config, "loss", -1.0)
            )
        else:
            final_reward = 0.0 if winner == 0 else -1.0

        legal_mask = self._as_bool_mask(getattr(self.env, "legal_actions_mask", None))
        transition = Transition(
            obs=obs,
            action=0,  # Dummy action, agent didn't actually move
            reward=final_reward,
            next_obs=opp_step.obs,
            terminated=opp_step.terminated,
            truncated=opp_step.truncated,
            info=opp_step.info,
            legal_mask=legal_mask if legal_mask is not None else np.zeros(self._get_action_dim(), dtype=bool),
            next_legal_mask=self._zero_mask_like(legal_mask),
        )
        return opp_step.obs, episode_length, transition, opp_step.info

    def _agent_turn(
        self,
        obs: np.ndarray,
        deterministic: bool,
    ) -> Tuple[StepResult, PendingTransition]:
        """Agent makes a move. Returns step result and pending transition."""
        legal = self._as_bool_mask(getattr(self.env, "legal_actions_mask", None))
        action = self.agent.act(obs, legal_mask=legal, deterministic=deterministic)
        step = self.env.step(action)
        pending = PendingTransition(obs=obs, action=action, reward=step.reward, legal_mask=legal)
        return step, pending  # step.obs is position before opponent's response

    def _opponent_turn_and_credit(
        self,
        obs_after_agent: np.ndarray,
        pending: PendingTransition,
        reward_config: Any,
    ) -> Tuple[StepResult, Transition]:
        """Opponent makes a move and create final transition for agent."""
        opp = self._current_opponent or self._prepare_opponent()
        opp_mask = self._as_bool_mask(getattr(self.env, "legal_actions_mask", None))
        opp_action = opp.act(obs_after_agent, legal_mask=opp_mask, deterministic=False)
        opp_step = self.env.step(opp_action)

        final_reward = self._compute_agent_reward(pending.reward, opp_step, reward_config)
        next_mask = (
            self._as_bool_mask(getattr(self.env, "legal_actions_mask", None))
            if not opp_step.done
            else self._zero_mask_like(pending.legal_mask)
        )
        transition = Transition(
            obs=pending.obs,
            action=pending.action,
            reward=final_reward,
            next_obs=opp_step.obs,
            terminated=opp_step.terminated,
            truncated=opp_step.truncated,
            info=opp_step.info,
            legal_mask=pending.legal_mask,
            next_legal_mask=next_mask,
        )
        return opp_step, transition

    def _handle_opponent_start(
        self,
        obs: np.ndarray,
        episode_reward: float,
        episode_length: int,
        reward_config: Any,
        iteration_start: Optional[float],
    ) -> Tuple[np.ndarray, float, int, bool]:
        """Handle opponent opening sequence when they move first."""
        obs, episode_length, transition, end_info = self._opponent_opening(obs, episode_length, reward_config)
        if transition is None:
            return obs, episode_reward, episode_length, True

        metrics = self._process_agent_transition(transition)
        episode_reward += transition.reward

        core_done = perf_counter() if self.track_timings else None
        core_start = iteration_start if iteration_start is not None else core_done
        self._after_transition(transition, metrics, core_start, core_done)

        obs, episode_reward, episode_length, pending, transition, end_info = self._handle_episode_end(
            episode_reward, episode_length, end_info or {}
        )
        iteration_start = perf_counter() if self.track_timings else None
        episode_reward = self._process_maybe_opening_transition(transition, episode_reward, iteration_start)
        current_player_is_agent = self._should_agent_go_first()
        return obs, episode_reward, episode_length, current_player_is_agent

    def _should_agent_go_first(self) -> bool:
        """Determine if agent should go first based on start policy."""
        if self.start_policy == StartPolicy.AGENT_FIRST:
            goes_first = True
        elif self.start_policy == StartPolicy.OPPONENT_FIRST:
            goes_first = False
        else:
            if isinstance(self._rng, random.Random):
                goes_first = self._rng.random() < 0.5
            else:  # np.random.Generator
                goes_first = self._rng.random() < 0.5
        self._agent_token = 1 if goes_first else -1
        return goes_first

    def _process_maybe_opening_transition(
        self,
        transition: Optional[Transition],
        episode_reward: float,
        iteration_start: Optional[float],
    ) -> float:
        """Process transition from opponent opening if present."""
        if transition is None:
            return episode_reward
        metrics = self._process_agent_transition(transition)
        episode_reward += transition.reward
        core_done = perf_counter() if self.track_timings else None
        self._after_transition(transition, metrics, iteration_start, core_done)
        return episode_reward

    def _apply_random_opening(self, initial_obs: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Apply randomized opening sequence if enabled."""
        cfg = self.random_opening_config
        if cfg is None:
            return initial_obs, False

        rand = self._rng if isinstance(self._rng, random.Random) else random

        if rand.random() >= cfg.probability:
            return initial_obs, False

        moves_to_play = rand.randint(cfg.min_half_moves, cfg.max_half_moves)
        obs = initial_obs
        done = False

        for _ in range(moves_to_play):
            if getattr(self.env, "done", False):
                done = True
                break
            legal_actions = self._get_legal_actions_for_opening()
            if not legal_actions:
                break
            action = rand.choice(legal_actions)
            step = self.env.step(action)
            if isinstance(step, StepResult):
                obs = step.obs
                done = step.done
            else:
                obs, _, done, _ = step
            if done:
                break

        return obs, done

    def _get_legal_actions_for_opening(self) -> List[int]:
        if hasattr(self.env, "get_legal_actions"):
            actions = self.env.get_legal_actions()
            return list(actions)
        mask = getattr(self.env, "legal_actions_mask", None)
        if mask is not None:
            mask_arr = np.asarray(mask, dtype=bool)
            return np.flatnonzero(mask_arr).tolist()
        raise RuntimeError("Environment must expose get_legal_actions() or legal_actions_mask for random opening.")

    def _agent_relative_result(self, winner: Optional[int]) -> Optional[int]:
        """Convert environment winner token into agent-centric result."""
        if winner is None:
            return None
        if winner == 0:
            return 0
        if winner == self._agent_token:
            return 1
        if winner == -self._agent_token:
            return -1
        return None

    def _compute_agent_reward(
        self,
        pending_reward: float,
        opponent_step: StepResult,
        reward_config: Any,
    ) -> float:
        """
        Compute final reward for agent's action after opponent's move.
        
        When opponent finishes the game (opponent_step.done == True):
        - If winner == 0: draw -> reward_config.draw
        - Otherwise: opponent won -> reward_config.loss (agent lost)
        
        When game continues: return pending_reward (shaping reward from agent's move).
        """
        if not opponent_step.done:
            # Game continues: return shaping reward from agent's move
            return pending_reward

        # Game ended after opponent's move
        winner = opponent_step.info.get("winner")
        if reward_config is not None:
            if winner == 0:
                # Draw
                return getattr(reward_config, "draw", 0.0)
            else:
                # Opponent won (regardless of whether winner == 1 or winner == -1)
                # Agent lost
                return getattr(reward_config, "loss", -1.0)

        # Fallback if no reward_config
        if winner == 0:
            return 0.0
        else:
            # Opponent won, agent lost
            return -1.0

    def _finalize_timings(self) -> None:
        total_time = self._timings["total"]
        core_time = self._timings["core"]
        callback_time = self._timings["callbacks"]
        eval_time = sum(
            getattr(cb, "time_spent", 0.0) for cb in self.callbacks if isinstance(cb, EvalCallback)
        )

        # Ensure non-negative residuals
        core_time = min(core_time, total_time)
        callback_time = min(callback_time, total_time)
        eval_time = min(eval_time, total_time)

        def pct(value: float) -> float:
            return (value / total_time * 100.0) if total_time > 0 else 0.0

        self.timing_report = {
            "total": total_time,
            "core": core_time,
            "callbacks": callback_time,
            "eval": eval_time,
        }

        print(
            "[timings] total={total:.2f}s, core={core:.2f}s ({core_pct:.1f}%), "
            "callbacks={cb:.2f}s ({cb_pct:.1f}%), eval={eval:.2f}s ({eval_pct:.1f}%)".format(
                total=total_time,
                core=core_time,
                core_pct=pct(core_time),
                cb=callback_time,
                cb_pct=pct(callback_time),
                eval=eval_time,
                eval_pct=pct(eval_time),
            )
        )

