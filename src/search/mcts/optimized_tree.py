"""Optimized batched MCTS with lazy child state creation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Generic, List, Optional, Tuple, TypeVar

import numpy as np

from src.games.turn_based_game import Action, TurnBasedGame

from .config import MCTSConfig
from .lazy_node import LazyMCTSNode

S = TypeVar("S")

BatchedEvaluator = Callable[
    [List[S]], Tuple[List[Dict[Action, float]], List[float]]
]


@dataclass
class MCTSSearchHandle(Generic[S]):
    """Tracks the state of an in-progress MCTS search."""

    root: LazyMCTSNode[S]
    num_sims: int
    sims_done: int = 0

    @property
    def is_done(self) -> bool:
        return self.sims_done >= self.num_sims


class OptimizedMCTS(Generic[S]):
    """
    Optimized MCTS with:
    - Batched neural network evaluation
    - Lazy child state creation (only create states when selected)
    - Stepped interface (begin_search / collect_leaves / apply_evaluations)
      for game-pool parallelism across K concurrent games
    """

    def __init__(
        self,
        game: TurnBasedGame[S],
        evaluator: BatchedEvaluator,
        config: MCTSConfig,
        rng: Optional[np.random.Generator] = None,
        batch_size: int = 32,
    ):
        self.game = game
        self.evaluator = evaluator
        self.config = config
        self.rng = rng or np.random.default_rng()
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Stepped interface (used by game pool in the trainer)
    # ------------------------------------------------------------------

    def begin_search(
        self,
        root_state: S,
        num_simulations: Optional[int] = None,
        add_dirichlet_noise: bool = True,
    ) -> MCTSSearchHandle[S]:
        """
        Initialise a new search tree from root_state.

        Evaluates the root state immediately (single forward pass) and
        returns a handle that tracks progress through the search.
        """
        num_sims = num_simulations or self.config.num_simulations

        root = LazyMCTSNode(state=root_state)
        root_priors, root_value = self._evaluate_single(root_state)
        self._expand_node_with_priors(root, root_priors)

        if add_dirichlet_noise and root._pending_actions:
            self._add_dirichlet_noise(root)

        root.backup(root_value)
        return MCTSSearchHandle(root=root, num_sims=num_sims)

    def collect_leaves(
        self, handle: MCTSSearchHandle[S]
    ) -> List[LazyMCTSNode[S]]:
        """
        Select up to batch_size leaves for one search step.

        Terminal leaves are handled immediately (backup their terminal value).
        Returns only the non-terminal leaves that need NN evaluation.
        Advances handle.sims_done by the full batch count.
        """
        batch_count = min(self.batch_size, handle.num_sims - handle.sims_done)
        non_terminal: List[LazyMCTSNode[S]] = []

        for _ in range(batch_count):
            leaf = self._select_leaf(handle.root)
            if leaf.is_terminal:
                leaf.backup(leaf.terminal_value)
            elif self.game.is_terminal(leaf.state):
                self._expand_node_with_priors(leaf, {})
                leaf.backup(leaf.terminal_value)
            else:
                non_terminal.append(leaf)
                self._apply_virtual_loss(leaf)

        handle.sims_done += batch_count
        return non_terminal

    def apply_evaluations(
        self,
        handle: MCTSSearchHandle[S],
        leaves: List[LazyMCTSNode[S]],
        priors_list: List[Dict[Action, float]],
        values_list: List[float],
    ) -> None:
        """Apply NN evaluation results to leaves and backup."""
        for leaf, priors, value in zip(leaves, priors_list, values_list):
            self._remove_virtual_loss(leaf)
            self._expand_node_with_priors(leaf, priors)
            if leaf.is_terminal:
                leaf.backup(leaf.terminal_value)
            else:
                leaf.backup(value)

    # ------------------------------------------------------------------
    # Original single-game interface (unchanged behaviour)
    # ------------------------------------------------------------------

    def search(
        self,
        root_state: S,
        num_simulations: Optional[int] = None,
        add_dirichlet_noise: bool = True,
    ) -> LazyMCTSNode[S]:
        handle = self.begin_search(root_state, num_simulations, add_dirichlet_noise)

        while not handle.is_done:
            leaves = self.collect_leaves(handle)
            if leaves:
                states = [leaf.state for leaf in leaves]
                all_priors, all_values = self.evaluator(states)
                self.apply_evaluations(handle, leaves, all_priors, all_values)

        return handle.root

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_leaf(self, root: LazyMCTSNode[S]) -> LazyMCTSNode[S]:
        node = root
        while node.is_expanded and not node.is_terminal:
            node = node.select_child(self.config.c_puct)
        return node

    def _apply_virtual_loss(self, leaf: LazyMCTSNode[S]) -> None:
        node = leaf
        while node is not None:
            node.visit_count += 1
            node.total_value += 1.0
            node = node.parent

    def _remove_virtual_loss(self, leaf: LazyMCTSNode[S]) -> None:
        node = leaf
        while node is not None:
            node.visit_count -= 1
            node.total_value -= 1.0
            node = node.parent

    def _evaluate_single(self, state: S) -> Tuple[Dict[Action, float], float]:
        all_priors, all_values = self.evaluator([state])
        return all_priors[0], all_values[0]

    def _expand_node_with_priors(
        self, node: LazyMCTSNode[S], priors: Dict[Action, float]
    ) -> None:
        state = node.state

        if self.game.is_terminal(state):
            node.is_terminal = True
            winner = self.game.winner(state)
            current_player = self.game.current_player(state)

            if winner is None or winner == 0:
                node.terminal_value = 0.0
            elif winner == current_player:
                node.terminal_value = 1.0
            else:
                node.terminal_value = -1.0
            return

        legal_actions = list(self.game.legal_actions(state))
        if not legal_actions:
            node.is_terminal = True
            node.terminal_value = 0.0
            return

        priors = self._normalize_priors(priors, legal_actions)
        node.expand_lazy(legal_actions, priors, self.game)

    def _normalize_priors(
        self,
        priors: Dict[Action, float],
        legal_actions: List[Action],
    ) -> Dict[Action, float]:
        # priors from the evaluator already has exactly the legal action keys,
        # so we skip the filter pass and normalize in-place.
        if not priors:
            n = len(legal_actions)
            return {a: 1.0 / n for a in legal_actions}

        total = sum(priors.values())
        if total <= 0:
            n = len(legal_actions)
            return {a: 1.0 / n for a in legal_actions}

        inv = 1.0 / total
        return {a: p * inv for a, p in priors.items()}

    def _add_dirichlet_noise(self, node: LazyMCTSNode[S]) -> None:
        alpha = self.config.dirichlet_alpha
        frac = self.config.dirichlet_frac

        if node._pending_actions and node._priors:
            actions = node._pending_actions
            noise = self.rng.dirichlet([alpha] * len(actions))

            for action, noise_val in zip(actions, noise):
                old_prior = node._priors.get(action, 0.0)
                node._priors[action] = (1 - frac) * old_prior + frac * noise_val

    def get_action_probs(
        self,
        root: LazyMCTSNode[S],
        temperature: float,
    ) -> Tuple[Action, Dict[Action, float]]:
        policy = root.get_policy_distribution(temperature)

        if temperature == 0.0:
            action = max(policy.keys(), key=lambda a: policy[a])
        else:
            actions = list(policy.keys())
            probs = [policy[a] for a in actions]
            action = self.rng.choice(actions, p=probs)

        return action, policy
