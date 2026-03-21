"""Optimized batched MCTS with lazy child state creation."""

from __future__ import annotations

from typing import Callable, Dict, Generic, List, Optional, Tuple, TypeVar

import numpy as np

from src.games.turn_based_game import Action, TurnBasedGame

from .config import MCTSConfig
from .lazy_node import LazyMCTSNode

S = TypeVar("S")

BatchedEvaluator = Callable[
    [List[S]], Tuple[List[Dict[Action, float]], List[float]]
]


class OptimizedMCTS(Generic[S]):
    """
    Optimized MCTS with:
    - Batched neural network evaluation
    - Lazy child state creation (only create states when selected)
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

    def search(
        self,
        root_state: S,
        num_simulations: Optional[int] = None,
        add_dirichlet_noise: bool = True,
    ) -> LazyMCTSNode[S]:
        num_sims = num_simulations or self.config.num_simulations

        root = LazyMCTSNode(state=root_state)

        root_priors, root_value = self._evaluate_single(root_state)
        self._expand_node_with_priors(root, root_priors)

        if add_dirichlet_noise and root._pending_actions:
            self._add_dirichlet_noise(root)

        root.backup(root_value)

        sims_done = 0
        while sims_done < num_sims:
            non_terminal_leaves = []
            terminal_leaves = []

            batch_count = min(self.batch_size, num_sims - sims_done)
            for _ in range(batch_count):
                leaf = self._select_leaf(root)
                if leaf.is_terminal:
                    # Already visited terminal: terminal_value is set, backup directly
                    terminal_leaves.append(leaf)
                elif self.game.is_terminal(leaf.state):
                    # First visit to a terminal: detect before sending to network
                    terminal_leaves.append(leaf)
                else:
                    non_terminal_leaves.append(leaf)
                    self._apply_virtual_loss(leaf)

            for leaf in terminal_leaves:
                if not leaf.is_terminal:
                    # First visit: set terminal value correctly
                    self._expand_node_with_priors(leaf, {})
                leaf.backup(leaf.terminal_value)

            if non_terminal_leaves:
                states = [leaf.state for leaf in non_terminal_leaves]
                all_priors, all_values = self.evaluator(states)

                for leaf, priors, value in zip(non_terminal_leaves, all_priors, all_values):
                    self._remove_virtual_loss(leaf)
                    self._expand_node_with_priors(leaf, priors)
                    # After expand, re-check: could be terminal on an edge case (no legal moves)
                    if leaf.is_terminal:
                        leaf.backup(leaf.terminal_value)
                    else:
                        leaf.backup(value)

            sims_done += batch_count

        return root

    def _select_leaf(self, root: LazyMCTSNode[S]) -> LazyMCTSNode[S]:
        """Select a leaf node using UCB."""
        node = root

        while node.is_expanded and not node.is_terminal:
            node = node.select_child(self.config.c_puct)

        return node

    def _apply_virtual_loss(self, leaf: LazyMCTSNode[S]) -> None:
        """Apply virtual loss along path from leaf to root to diversify batch selection."""
        node = leaf
        while node is not None:
            node.visit_count += 1
            # Positive value makes Q positive, so -Q in UCB becomes negative
            # This discourages selecting this path again
            node.total_value += 1.0
            node = node.parent

    def _remove_virtual_loss(self, leaf: LazyMCTSNode[S]) -> None:
        """Remove virtual loss before applying real backup."""
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
        legal_priors = {a: priors.get(a, 0.0) for a in legal_actions}
        total = sum(legal_priors.values())

        if total <= 0:
            n = len(legal_actions)
            return {a: 1.0 / n for a in legal_actions}

        return {a: p / total for a, p in legal_priors.items()}

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
