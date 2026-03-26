"""AlphaZero trainer implementation."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

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

    # Fraction of fresh self-play samples withheld from replay buffer for
    # validation loss tracking.  0.0 disables the val split.
    val_split_fraction: float = 0.0

    # Re-search stability: run MCTS twice (independent RNG + Dirichlet) on a few
    # root states from self-play; 0 disables. Low counts keep overhead small.
    re_search_stability_positions: int = 0
    # If set, overrides num simulations for these diagnostic searches only.
    re_search_stability_num_sims: Optional[int] = None


_RESEARCH_STABILITY_POOL_CAP = 256


def _symmetric_kl_discrete(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    kl_pq = np.sum(np.where(p > 0, p * (np.log(p + eps) - np.log(q + eps)), 0.0))
    kl_qp = np.sum(np.where(q > 0, q * (np.log(q + eps) - np.log(p + eps)), 0.0))
    return 0.5 * (float(kl_pq) + float(kl_qp))


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
        effective_batch = config.mcts_batch_size * min(config.game_pool_size, config.num_games_per_iteration)
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

        # Held-out samples from the most recent self-play phase (not in replay buffer).
        self._val_samples: Optional[List[AlphaZeroSample]] = None

        self._stability_state_pool: List[Any] = []

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

        # Compute search quality on fresh samples while network is in eval mode.
        search_metrics = self._compute_search_quality(samples)
        research_metrics = self._compute_re_search_stability()

        for cb in self.callbacks:
            cb.on_self_play_end(
                self,
                self.iteration,
                self.config.num_games_per_iteration,
                len(samples),
            )

        # Optionally withhold a fraction of fresh samples for validation (RNG indices).
        if self.config.val_split_fraction > 0.0 and samples:
            n_val = max(1, int(len(samples) * self.config.val_split_fraction))
            n_val = min(n_val, len(samples))
            val_indices = self.rng.choice(len(samples), size=n_val, replace=False)
            val_set = set(int(i) for i in val_indices)
            self._val_samples = [samples[i] for i in val_indices]
            train_samples = [samples[i] for i in range(len(samples)) if i not in val_set]
        else:
            self._val_samples = None
            train_samples = samples

        for sample in train_samples:
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

        # Val losses: network is in eval mode at this point.
        val_metrics = self._compute_val_losses()

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
            **search_metrics,
            **research_metrics,
            **val_metrics,
        }

        for cb in self.callbacks:
            cb.on_iteration_end(self, self.iteration, iteration_metrics)

    def _self_play_phase(self) -> List[AlphaZeroSample]:
        self.agent.eval()
        self._stability_state_pool.clear()

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

        effective_K = min(K, total_needed)
        slots: List[Optional[_GameSlot]] = [_make_slot() for _ in range(effective_K)]

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

                self._maybe_record_re_search_root_state(slot.state)

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

            self._maybe_record_re_search_root_state(state)
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
        policy_loss_trace: List[float] = []
        value_loss_trace: List[float] = []
        max_kl = self.config.max_kl_divergence
        kl_active = (
            max_kl is not None
            and self.iteration > self.config.kl_warmup_iterations
        )

        # Snapshot BN running stats from before training starts.
        # KL probes always use these stats so we measure weight-only change,
        # not the conflated (weight + BN running stats drift) artifact.
        bn_stats_before = self._snapshot_bn_stats() if kl_active else {}

        # Fixed probe batch for per-step loss traces.  Sampled once so all
        # steps see the same positions, making the trace directly comparable.
        _PROBE_SIZE = 512
        probe_batch = (
            self.agent.replay_buffer.sample(_PROBE_SIZE, self.agent.device)
            if len(self.agent.replay_buffer) >= _PROBE_SIZE
            else None
        )

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

            # Per-step probe losses (after weight update so trace[t] = post-step t+1).
            if probe_batch is not None:
                p_l, v_l = self._eval_probe_losses(probe_batch)
                policy_loss_trace.append(p_l)
                value_loss_trace.append(v_l)
                self.agent.train()

        if steps == 0:
            return {
                "kl_stopped": kl_stopped,
                "kl_trace": kl_trace,
                "policy_loss_trace": policy_loss_trace,
                "value_loss_trace": value_loss_trace,
            }

        age = self.agent.replay_buffer.age_stats(self.iteration, new_samples)
        return {
            "avg_loss": total_loss / steps,
            "avg_policy_loss": total_policy_loss / steps,
            "avg_value_loss": total_value_loss / steps,
            "train_steps": steps,
            "kl_stopped": kl_stopped,
            "kl_trace": kl_trace,
            "policy_loss_trace": policy_loss_trace,
            "value_loss_trace": value_loss_trace,
            **age,
        }

    def _maybe_record_re_search_root_state(self, state: Any) -> None:
        if self.config.re_search_stability_positions <= 0:
            return
        if len(self._stability_state_pool) >= _RESEARCH_STABILITY_POOL_CAP:
            return
        self._stability_state_pool.append(copy.deepcopy(state))

    def _root_visit_policy_array(self, root: Any) -> np.ndarray:
        dist = root.get_policy_distribution(1.0)
        p = np.zeros(self.agent.num_actions, dtype=np.float64)
        for a, prob in dist.items():
            p[int(a)] = float(prob)
        tot = float(p.sum())
        if tot > 0:
            p /= tot
        return p

    def _compute_re_search_stability(self) -> Dict[str, Any]:
        """
        Two full MCTS runs (independent RNG / Dirichlet) per sampled root from self-play.
        Measures search noise: symmetric KL of visit policies, top-1 agreement, |ΔQ_root|.
        """
        npos = self.config.re_search_stability_positions
        if npos <= 0 or not self._stability_state_pool:
            return {}

        pool = self._stability_state_pool
        k = min(npos, len(pool))
        pick = self.rng.choice(len(pool), size=k, replace=False)

        num_sims = self.config.re_search_stability_num_sims
        if num_sims is None:
            num_sims = self.config.mcts_simulations

        sym_kls: List[float] = []
        top1s: List[float] = []
        vdiffs: List[float] = []

        for j in pick:
            state_snapshot = pool[int(j)]
            rng1 = np.random.default_rng(int(self.rng.integers(1 << 63)))
            rng2 = np.random.default_rng(int(self.rng.integers(1 << 63)))
            m1 = OptimizedMCTS(
                self.game,
                self._batched_evaluator,
                self._mcts_config,
                rng=rng1,
                batch_size=self.config.mcts_batch_size,
            )
            m2 = OptimizedMCTS(
                self.game,
                self._batched_evaluator,
                self._mcts_config,
                rng=rng2,
                batch_size=self.config.mcts_batch_size,
            )
            r1 = m1.search(
                copy.deepcopy(state_snapshot),
                num_simulations=num_sims,
                add_dirichlet_noise=True,
            )
            r2 = m2.search(
                copy.deepcopy(state_snapshot),
                num_simulations=num_sims,
                add_dirichlet_noise=True,
            )
            p1 = self._root_visit_policy_array(r1)
            p2 = self._root_visit_policy_array(r2)
            sym_kls.append(_symmetric_kl_discrete(p1, p2))
            top1s.append(1.0 if int(np.argmax(p1)) == int(np.argmax(p2)) else 0.0)
            vdiffs.append(abs(float(r1.q_value) - float(r2.q_value)))

        return {
            "policy_research_kl": float(np.mean(sym_kls)),
            "policy_research_top1": float(np.mean(top1s)),
            "value_research_absdiff": float(np.mean(vdiffs)),
            "re_search_n": k,
        }

    def _compute_search_quality(self, samples: List[AlphaZeroSample]) -> Dict[str, Any]:
        """
        KL and agreement metrics between the MCTS policy and the network prior.

        All positions in *samples* are evaluated together.  Metrics:
            search_kl            - KL(π_mcts || p_net)  average over positions
            top1_agreement       - fraction where argmax(p_net) == argmax(π_mcts)
            mass_on_best         - p_net probability assigned to MCTS best move
            target_entropy       - H(π_mcts) average over positions
            target_entropy_norm  - H(π_mcts) / log(|A_legal|), average
        """
        if not samples:
            return {}

        obs_np = np.stack([s.observation for s in samples])
        mask_np = np.stack([s.legal_mask for s in samples]).astype(bool)
        pi_np = np.stack([s.target_policy for s in samples])

        obs_t = torch.as_tensor(obs_np, device=self.agent.device)
        mask_t = torch.as_tensor(mask_np, device=self.agent.device)

        with torch.inference_mode():
            p_net_t, _ = self.agent.network.predict(obs_t, mask_t)

        p_net = p_net_t.cpu().numpy()

        eps = 1e-8
        # KL(π || p_net) — only over actions where π > 0
        kl_per = np.where(
            pi_np > 0,
            pi_np * (np.log(np.clip(pi_np, eps, None)) - np.log(np.clip(p_net, eps, None))),
            0.0,
        ).sum(axis=-1)
        search_kl = float(kl_per.mean())

        best_mcts = np.argmax(pi_np, axis=-1)
        best_net = np.argmax(p_net, axis=-1)
        top1_agreement = float((best_net == best_mcts).mean())

        n = len(samples)
        mass_on_best = float(p_net[np.arange(n), best_mcts].mean())

        h_per = -np.where(
            pi_np > 0,
            pi_np * np.log(np.clip(pi_np, eps, None)),
            0.0,
        ).sum(axis=-1)
        target_entropy = float(h_per.mean())

        n_legal = mask_np.sum(axis=-1).astype(float)
        max_h = np.log(np.maximum(n_legal, 1.0))
        # Positions with one legal move have max_h=0 and h_per=0; normalized entropy = 0.
        target_entropy_norm = float(
            np.divide(h_per, max_h, out=np.zeros_like(h_per), where=max_h > 0).mean()
        )

        return {
            "search_kl": search_kl,
            "top1_agreement": top1_agreement,
            "mass_on_best": mass_on_best,
            "target_entropy": target_entropy,
            "target_entropy_norm": target_entropy_norm,
        }

    def _compute_val_losses(self) -> Dict[str, Any]:
        """
        Cross-entropy policy loss and MSE value loss on the held-out val set.
        Returns an empty dict when val_split_fraction == 0 or no val samples exist.
        """
        if not self._val_samples:
            return {}

        obs_np = np.stack([s.observation for s in self._val_samples])
        mask_np = np.stack([s.legal_mask for s in self._val_samples]).astype(bool)
        pi_np = np.stack([s.target_policy for s in self._val_samples]).astype(np.float32)
        z_np = np.array([s.target_value for s in self._val_samples], dtype=np.float32)

        dev = self.agent.device
        obs_t = torch.as_tensor(obs_np, device=dev)
        mask_t = torch.as_tensor(mask_np, device=dev)
        pi_t = torch.as_tensor(pi_np, device=dev)
        z_t = torch.as_tensor(z_np, device=dev).unsqueeze(-1)

        with torch.inference_mode():
            policy_logits, value_pred = self.agent.network(obs_t, legal_mask=None)
            masked_logits = policy_logits.masked_fill(~mask_t, -1e9)
            log_probs = torch.log_softmax(masked_logits, dim=-1)
            val_policy_loss = -(pi_t * log_probs).clamp(min=-100.0).sum(dim=-1).mean()
            val_value_loss = torch.nn.functional.mse_loss(value_pred, z_t)

        return {
            "val_policy_loss": float(val_policy_loss),
            "val_value_loss": float(val_value_loss),
        }

    def _eval_probe_losses(self, probe_batch) -> Tuple[float, float]:
        """Policy cross-entropy and value MSE on a fixed probe batch."""
        self.agent.eval()
        with torch.inference_mode():
            policy_logits, value_pred = self.agent.network(
                probe_batch.observations, legal_mask=None
            )
            masked_logits = policy_logits.masked_fill(~probe_batch.legal_masks, -1e9)
            log_probs = torch.log_softmax(masked_logits, dim=-1)
            p_loss = -(probe_batch.target_policies * log_probs).clamp(min=-100.0).sum(dim=-1).mean()
            v_loss = torch.nn.functional.mse_loss(value_pred, probe_batch.target_values)
        return float(p_loss), float(v_loss)

    def _save_checkpoint(self) -> None:
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        path = checkpoint_dir / f"alphazero_iter_{self.iteration:06d}.pt"
        self.agent.save(str(path))
        print(f"Saved checkpoint: {path}")
