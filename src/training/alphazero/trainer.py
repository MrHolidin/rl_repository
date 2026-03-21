"""AlphaZero trainer implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar

import numpy as np

from src.agents.alphazero.agent import AlphaZeroAgent
from src.agents.alphazero.replay_buffer import AlphaZeroSample
from src.features.observation_builder import ObservationBuilder
from src.games.turn_based_game import Action, TurnBasedGame
from src.search.mcts.config import MCTSConfig
from src.search.mcts.optimized_tree import OptimizedMCTS
from src.search.mcts.batched_evaluator import make_batched_evaluator
from src.search.mcts.cuda_graph_evaluator import make_cuda_graph_evaluator

S = TypeVar("S")


@dataclass
class AlphaZeroConfig:
    """Configuration for AlphaZero training."""

    num_games_per_iteration: int = 25
    mcts_simulations: int = 100
    mcts_c_puct: float = 1.4
    mcts_batch_size: int = 48
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25
    temperature: float = 1.0
    temperature_threshold: int = 15

    num_iterations: int = 10
    train_steps_per_iteration: int = 100
    batch_size: int = 256
    learning_rate: float = 0.002
    weight_decay: float = 1e-4

    checkpoint_interval: int = 5
    checkpoint_dir: str = "checkpoints"

    eval_interval: int = 5
    eval_games: int = 10
    
    use_cuda_graph: bool = False


class AlphaZeroTrainerCallback:
    """Base callback for AlphaZero trainer."""

    def on_iteration_start(self, trainer: "AlphaZeroTrainer", iteration: int) -> None:
        pass

    def on_self_play_end(
        self,
        trainer: "AlphaZeroTrainer",
        iteration: int,
        games_played: int,
        samples_collected: int,
    ) -> None:
        pass

    def on_training_step(
        self,
        trainer: "AlphaZeroTrainer",
        iteration: int,
        step: int,
        metrics: Dict[str, float],
    ) -> None:
        pass

    def on_iteration_end(
        self,
        trainer: "AlphaZeroTrainer",
        iteration: int,
        metrics: Dict[str, Any],
    ) -> None:
        pass


@dataclass
class GameSample:
    """Intermediate sample before game outcome is known."""

    observation: np.ndarray
    legal_mask: np.ndarray
    mcts_policy: np.ndarray
    player_token: int


class AlphaZeroTrainer:
    """
    AlphaZero training loop.

    Training cycle:
    1. Self-play: Generate games using MCTS + current network
    2. Training: Update network on collected samples
    """

    def __init__(
        self,
        game: TurnBasedGame,
        agent: AlphaZeroAgent,
        observation_builder: ObservationBuilder,
        state_to_dict_fn: Callable,
        initial_state_fn: Callable,
        config: AlphaZeroConfig,
        callbacks: Optional[List[AlphaZeroTrainerCallback]] = None,
        augment_fn: Optional[
            Callable[[AlphaZeroSample], Sequence[AlphaZeroSample]]
        ] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.game = game
        self.agent = agent
        self.observation_builder = observation_builder
        self.state_to_dict_fn = state_to_dict_fn
        self.initial_state_fn = initial_state_fn
        self.config = config
        self.callbacks = callbacks or []
        self.augment_fn = augment_fn
        self.rng = rng or np.random.default_rng()

        self._mcts_config = MCTSConfig(
            num_simulations=config.mcts_simulations,
            c_puct=config.mcts_c_puct,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_frac=config.dirichlet_frac,
            temperature=config.temperature,
            temperature_threshold=config.temperature_threshold,
        )

        if config.use_cuda_graph and str(agent.device) != "cpu":
            self._batched_evaluator = make_cuda_graph_evaluator(
                agent, game, state_to_dict_fn, observation_builder,
                fixed_batch_size=config.mcts_batch_size,
            )
        else:
            self._batched_evaluator = make_batched_evaluator(
                agent, game, state_to_dict_fn, observation_builder
            )

        self.iteration = 0
        self.total_games = 0
        self.total_samples = 0

    def train(self, num_iterations: Optional[int] = None) -> None:
        """Run the training loop."""
        num_iters = num_iterations or self.config.num_iterations

        for _ in range(num_iters):
            self._run_iteration()

    def _run_iteration(self) -> None:
        """Run a single training iteration."""
        self.iteration += 1
        iter_start = perf_counter()

        for cb in self.callbacks:
            cb.on_iteration_start(self, self.iteration)

        self_play_start = perf_counter()
        samples = self._self_play_phase()
        self_play_time = perf_counter() - self_play_start

        for cb in self.callbacks:
            cb.on_self_play_end(
                self,
                self.iteration,
                self.config.num_games_per_iteration,
                len(samples),
            )

        for sample in samples:
            self.agent.replay_buffer.push(sample)
            if self.augment_fn is not None:
                for aug_sample in self.augment_fn(sample):
                    self.agent.replay_buffer.push(aug_sample)

        train_start = perf_counter()
        train_metrics = self._training_phase()
        train_time = perf_counter() - train_start

        if (
            self.config.checkpoint_interval > 0
            and self.iteration % self.config.checkpoint_interval == 0
        ):
            self._save_checkpoint()

        iter_time = perf_counter() - iter_start

        iteration_metrics = {
            "games_played": self.config.num_games_per_iteration,
            "samples_collected": len(samples),
            "buffer_size": len(self.agent.replay_buffer),
            "self_play_time": self_play_time,
            "train_time": train_time,
            "iteration_time": iter_time,
            **train_metrics,
        }

        for cb in self.callbacks:
            cb.on_iteration_end(self, self.iteration, iteration_metrics)

    def _self_play_phase(self) -> List[AlphaZeroSample]:
        """Generate self-play games and collect samples."""
        self.agent.eval()

        all_samples: List[AlphaZeroSample] = []

        for _ in range(self.config.num_games_per_iteration):
            samples = self._play_one_game()
            all_samples.extend(samples)
            self.total_games += 1

        self.total_samples += len(all_samples)
        return all_samples

    def _play_one_game(self) -> List[AlphaZeroSample]:
        """Play a single self-play game and return samples."""
        state = self.initial_state_fn()
        game_samples: List[GameSample] = []
        move_number = 0

        mcts = OptimizedMCTS(
            self.game,
            self._batched_evaluator,
            self._mcts_config,
            self.rng,
            batch_size=self.config.mcts_batch_size,
        )

        while not self.game.is_terminal(state):
            current_player = self.game.current_player(state)

            root = mcts.search(state, add_dirichlet_noise=True)

            temperature = self._mcts_config.get_temperature(move_number)
            action, policy_dict = mcts.get_action_probs(root, temperature)

            policy_array = np.zeros(self.agent.num_actions, dtype=np.float32)
            for a, p in policy_dict.items():
                policy_array[a] = p

            obs = self._state_to_obs(state)
            legal_mask = self._get_legal_mask(state)

            game_samples.append(
                GameSample(
                    observation=obs,
                    legal_mask=legal_mask,
                    mcts_policy=policy_array,
                    player_token=current_player,
                )
            )

            state = self.game.apply_action(state, action)
            move_number += 1

        winner = self.game.winner(state)

        final_samples: List[AlphaZeroSample] = []

        for gs in game_samples:
            if winner == 0:
                value = 0.0
            elif winner == gs.player_token:
                value = 1.0
            else:
                value = -1.0

            final_samples.append(
                AlphaZeroSample(
                    observation=gs.observation,
                    legal_mask=gs.legal_mask,
                    target_policy=gs.mcts_policy,
                    target_value=value,
                )
            )

        return final_samples

    def _state_to_obs(self, state) -> np.ndarray:
        """Convert game state to observation."""
        legal_mask = self._get_legal_mask(state)
        state_dict = self.state_to_dict_fn(state, self.game, legal_mask=legal_mask)
        return self.observation_builder.build(state_dict)

    def _get_legal_mask(self, state) -> np.ndarray:
        """Get legal action mask for state."""
        legal_actions = list(self.game.legal_actions(state))
        mask = np.zeros(self.agent.num_actions, dtype=bool)
        for a in legal_actions:
            mask[a] = True
        return mask

    def _training_phase(self) -> Dict[str, float]:
        """Train the network on collected samples."""
        self.agent.train()

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        steps = 0

        for step in range(self.config.train_steps_per_iteration):
            metrics = self.agent.update()

            if metrics:
                total_loss += metrics.get("loss", 0.0)
                total_policy_loss += metrics.get("policy_loss", 0.0)
                total_value_loss += metrics.get("value_loss", 0.0)
                steps += 1

                for cb in self.callbacks:
                    cb.on_training_step(self, self.iteration, step, metrics)

        if steps == 0:
            return {}

        return {
            "avg_loss": total_loss / steps,
            "avg_policy_loss": total_policy_loss / steps,
            "avg_value_loss": total_value_loss / steps,
            "train_steps": steps,
        }

    def _save_checkpoint(self) -> None:
        """Save checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        path = checkpoint_dir / f"alphazero_iter_{self.iteration:06d}.pt"
        self.agent.save(str(path))
        print(f"Saved checkpoint: {path}")
