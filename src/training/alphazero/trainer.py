"""AlphaZero trainer implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar

import numpy as np
import torch

from src.agents.alphazero.agent import AlphaZeroAgent
from src.agents.alphazero.replay_buffer import AlphaZeroSample
from src.features.observation_builder import ObservationBuilder
from src.games.turn_based_game import Action, TurnBasedGame
from src.search.mcts.config import MCTSConfig
from src.search.mcts.optimized_tree import MCTSSearchHandle, OptimizedMCTS
from src.search.mcts.batched_evaluator import make_batched_evaluator
from src.search.mcts.cuda_graph_evaluator import make_cuda_graph_evaluator

S = TypeVar("S")


@dataclass
class AlphaZeroConfig:
    """Configuration for AlphaZero training."""

    num_games_per_iteration: int = 25
    mcts_simulations: int = 100
    mcts_c_puct: float = 1.4
    mcts_batch_size: int = 256
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25
    temperature: float = 1.0
    temperature_threshold: int = 15

    num_iterations: int = 10
    max_train_steps_per_iteration: int = 100
    max_kl_divergence: Optional[float] = None
    kl_warmup_iterations: int = 3
    learning_rate: float = 0.002
    weight_decay: float = 1e-4

    checkpoint_interval: int = 5
    checkpoint_dir: str = "checkpoints"

    use_cuda_graph: bool = False
    game_pool_size: int = 1


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


@dataclass
class _GameSlot:
    """Active game context in the game pool."""

    state: Any
    mcts: OptimizedMCTS
    handle: MCTSSearchHandle
    game_samples: List[GameSample]
    move_number: int


class AlphaZeroTrainer:
    """
    AlphaZero training loop.

    Training cycle:
    1. Self-play: Generate games using MCTS + current network
    2. Training: Update network on collected samples

    When config.game_pool_size > 1, self-play runs K games concurrently,
    batching NN evaluations across all active games for better GPU utilisation.
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

        # CUDA graph batch covers all K games simultaneously for leaf evaluation.
        # Root evaluations (single state per begin_search) use a separate small
        # evaluator so they don't go through a batch=K*N CUDA graph.
        effective_batch = config.mcts_batch_size * config.game_pool_size
        if config.use_cuda_graph and str(agent.device) != "cpu":
            self._batched_evaluator = make_cuda_graph_evaluator(
                agent, game, state_to_dict_fn, observation_builder,
                fixed_batch_size=effective_batch,
            )
            if config.game_pool_size > 1:
                # Small evaluator for single-state root evals in begin_search
                self._root_evaluator = make_batched_evaluator(
                    agent, game, state_to_dict_fn, observation_builder
                )
            else:
                self._root_evaluator = self._batched_evaluator
        else:
            self._batched_evaluator = make_batched_evaluator(
                agent, game, state_to_dict_fn, observation_builder
            )
            self._root_evaluator = self._batched_evaluator

        self.iteration = 0
        self.total_games = 0
        self.total_samples = 0

        self._kl_probe_obs: Optional[np.ndarray] = None
        self._kl_probe_masks: Optional[np.ndarray] = None

    def train(self, num_iterations: Optional[int] = None) -> None:
        """Run the training loop."""
        num_iters = num_iterations or self.config.num_iterations

        for _ in range(num_iters):
            self._run_iteration()

    def _run_iteration(self) -> None:
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
            self.agent.replay_buffer.push(sample, iteration=self.iteration)
            if self.augment_fn is not None:
                for aug_sample in self.augment_fn(sample):
                    self.agent.replay_buffer.push(aug_sample, iteration=self.iteration)

        self.agent.eval()
        policy_before = self._get_probe_policy()
        bn_stats_before = self._snapshot_bn_stats()

        train_start = perf_counter()
        train_metrics = self._training_phase(new_samples=len(samples), policy_before=policy_before)
        train_time = perf_counter() - train_start

        if policy_before is not None:
            # Restore S₀ so policy_after uses the same BN normalization as policy_before.
            current_bn = self._snapshot_bn_stats()
            self._restore_bn_stats(bn_stats_before)
            self.agent.eval()
            policy_after = self._get_probe_policy()
            self.agent.train()
            self._restore_bn_stats(current_bn)
            self.agent.eval()
            kl = (policy_after * (
                np.log(policy_after + 1e-8) - np.log(policy_before + 1e-8)
            )).sum(axis=-1).mean()
            train_metrics["policy_kl"] = float(kl)

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
        self.agent.eval()

        if self.config.game_pool_size > 1:
            return self._self_play_phase_pooled()

        all_samples: List[AlphaZeroSample] = []
        for _ in range(self.config.num_games_per_iteration):
            samples = self._play_one_game()
            all_samples.extend(samples)
            self.total_games += 1

        self.total_samples += len(all_samples)
        return all_samples

    def _self_play_phase_pooled(self) -> List[AlphaZeroSample]:
        """
        Run config.game_pool_size games concurrently.

        Each loop iteration:
          1. Collect leaves from all active MCTS searches.
          2. Single evaluator call for all non-terminal leaves combined.
          3. Apply results back to each game's search handle.
          4. For completed searches: take action, advance game state,
             and start the next search (or finalise if terminal).
        """
        K = self.config.game_pool_size
        total_needed = self.config.num_games_per_iteration
        games_started = 0
        games_completed = 0
        all_samples: List[AlphaZeroSample] = []

        def _make_slot() -> Optional[_GameSlot]:
            nonlocal games_started
            if games_started >= total_needed:
                return None
            state = self.initial_state_fn()
            # Root evaluator is small (no large CUDA graph overhead for batch=1).
            # Leaf evaluations are batched across all K games by the pool loop directly.
            mcts = OptimizedMCTS(
                self.game,
                self._root_evaluator,
                self._mcts_config,
                self.rng,
                batch_size=self.config.mcts_batch_size,
            )
            handle = mcts.begin_search(state, add_dirichlet_noise=True)
            games_started += 1
            return _GameSlot(
                state=state,
                mcts=mcts,
                handle=handle,
                game_samples=[],
                move_number=0,
            )

        slots: List[Optional[_GameSlot]] = [_make_slot() for _ in range(K)]

        while games_completed < total_needed:
            # --- 1. Collect leaves from all active slots ---
            all_leaves: List = []
            slot_leaf_counts: List[int] = []

            for slot in slots:
                if slot is None or slot.handle.is_done:
                    slot_leaf_counts.append(0)
                    continue
                leaves = slot.mcts.collect_leaves(slot.handle)
                all_leaves.extend(leaves)
                slot_leaf_counts.append(len(leaves))

            # --- 2. Single batched NN evaluation ---
            if all_leaves:
                states = [leaf.state for leaf in all_leaves]
                all_priors, all_values = self._batched_evaluator(states)

                # --- 3. Distribute results back to each slot ---
                offset = 0
                for slot, count in zip(slots, slot_leaf_counts):
                    if count > 0:
                        slot.mcts.apply_evaluations(
                            slot.handle,
                            all_leaves[offset : offset + count],
                            all_priors[offset : offset + count],
                            all_values[offset : offset + count],
                        )
                        offset += count

            # --- 4. Advance slots where MCTS search is complete ---
            for i, slot in enumerate(slots):
                if slot is None or not slot.handle.is_done:
                    continue

                temperature = self._mcts_config.get_temperature(slot.move_number)
                action, policy_dict = slot.mcts.get_action_probs(
                    slot.handle.root, temperature
                )

                policy_array = np.zeros(self.agent.num_actions, dtype=np.float32)
                for a, p in policy_dict.items():
                    policy_array[a] = p

                slot.game_samples.append(
                    GameSample(
                        observation=self._state_to_obs(slot.state),
                        legal_mask=self._get_legal_mask(slot.state),
                        mcts_policy=policy_array,
                        player_token=self.game.current_player(slot.state),
                    )
                )

                new_state = self.game.apply_action(slot.state, action)

                if self.game.is_terminal(new_state):
                    winner = self.game.winner(new_state)
                    for gs in slot.game_samples:
                        if winner == 0:
                            value = 0.0
                        elif winner == gs.player_token:
                            value = 1.0
                        else:
                            value = -1.0
                        all_samples.append(
                            AlphaZeroSample(
                                observation=gs.observation,
                                legal_mask=gs.legal_mask,
                                target_policy=gs.mcts_policy,
                                target_value=value,
                            )
                        )
                    games_completed += 1
                    self.total_games += 1
                    slots[i] = _make_slot()
                else:
                    slot.state = new_state
                    slot.move_number += 1
                    slot.handle = slot.mcts.begin_search(
                        new_state, add_dirichlet_noise=True
                    )

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
        state_dict = self.state_to_dict_fn(state, self.game)
        return self.observation_builder.build(state_dict)

    def _get_legal_mask(self, state) -> np.ndarray:
        legal_actions = list(self.game.legal_actions(state))
        mask = np.zeros(self.agent.num_actions, dtype=bool)
        for a in legal_actions:
            mask[a] = True
        return mask

    _KL_PROBE_SIZE = 512

    def _get_probe_policy(self) -> Optional[np.ndarray]:
        """Return policy probabilities on fixed probe positions. Initialises probes on first call."""
        buf = self.agent.replay_buffer
        if self._kl_probe_obs is None:
            if len(buf) < self._KL_PROBE_SIZE:
                return None
            batch = buf.sample(self._KL_PROBE_SIZE, torch.device("cpu"))
            self._kl_probe_obs = batch.observations.numpy()
            self._kl_probe_masks = batch.legal_masks.numpy()

        obs_t  = torch.as_tensor(self._kl_probe_obs,   device=self.agent.device)
        mask_t = torch.as_tensor(self._kl_probe_masks, device=self.agent.device)
        with torch.inference_mode():
            policy, _ = self.agent.network.predict(obs_t, mask_t)
        return policy.cpu().numpy()

    def _snapshot_bn_stats(self) -> dict:
        """Save BatchNorm running_mean/running_var for all BN layers."""
        snapshot = {}
        for name, module in self.agent.network.named_modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                snapshot[name] = (module.running_mean.clone(), module.running_var.clone())
        return snapshot

    def _restore_bn_stats(self, snapshot: dict) -> None:
        """Restore previously snapshotted BatchNorm running stats."""
        for name, module in self.agent.network.named_modules():
            if name in snapshot:
                module.running_mean.copy_(snapshot[name][0])
                module.running_var.copy_(snapshot[name][1])

    def _training_phase(
        self,
        new_samples: int = 0,
        policy_before: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        self.agent.train()

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        steps = 0
        kl_stopped = False
        kl_trace: List[float] = []
        max_kl = self.config.max_kl_divergence
        kl_active = (
            max_kl is not None
            and self.iteration > self.config.kl_warmup_iterations
        )

        # Snapshot BN running stats from before training starts.
        # KL probes always use these stats so we measure weight-only change,
        # not the conflated (weight + BN running stats drift) artifact.
        bn_stats_before = self._snapshot_bn_stats() if kl_active else {}

        for step in range(self.config.max_train_steps_per_iteration):
            if kl_active and policy_before is not None:
                # Temporarily restore pre-training BN stats so the probe sees
                # the same normalization as policy_before was computed under.
                current_bn = self._snapshot_bn_stats()
                self._restore_bn_stats(bn_stats_before)
                self.agent.eval()
                policy_now = self._get_probe_policy()
                self.agent.train()
                self._restore_bn_stats(current_bn)
                if policy_now is not None:
                    kl = float(
                        (policy_now * (
                            np.log(policy_now + 1e-8) - np.log(policy_before + 1e-8)
                        )).sum(axis=-1).mean()
                    )
                    kl_trace.append(kl)
                    if kl > max_kl:
                        kl_stopped = True
                        break

            metrics = self.agent.update()

            if metrics:
                total_loss += metrics.get("loss", 0.0)
                total_policy_loss += metrics.get("policy_loss", 0.0)
                total_value_loss += metrics.get("value_loss", 0.0)
                steps += 1

                for cb in self.callbacks:
                    cb.on_training_step(self, self.iteration, step, metrics)

        if steps == 0:
            return {"kl_stopped": kl_stopped, "kl_trace": kl_trace}

        age = self.agent.replay_buffer.age_stats(self.iteration, new_samples)
        return {
            "avg_loss": total_loss / steps,
            "avg_policy_loss": total_policy_loss / steps,
            "avg_value_loss": total_value_loss / steps,
            "train_steps": steps,
            "kl_stopped": kl_stopped,
            "kl_trace": kl_trace,
            **age,
        }

    def _save_checkpoint(self) -> None:
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        path = checkpoint_dir / f"alphazero_iter_{self.iteration:06d}.pt"
        self.agent.save(str(path))
        print(f"Saved checkpoint: {path}")
