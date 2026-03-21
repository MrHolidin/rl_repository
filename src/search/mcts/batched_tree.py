"""Batched MCTS with leaf parallelization for GPU efficiency."""

from __future__ import annotations

from typing import Callable, Dict, Generic, List, Optional, Tuple, TypeVar

import numpy as np
import torch

from src.games.turn_based_game import Action, TurnBasedGame

from .config import MCTSConfig
from .node import MCTSNode

S = TypeVar("S")

BatchedEvaluator = Callable[
    [List[S]], Tuple[List[Dict[Action, float]], List[float]]
]


class BatchedMCTS(Generic[S]):
    """
    MCTS with batched neural network evaluation.
    
    Instead of evaluating one leaf at a time, accumulates multiple leaves
    and evaluates them in a single batched forward pass.
    """

    def __init__(
        self,
        game: TurnBasedGame[S],
        evaluator: BatchedEvaluator,
        config: MCTSConfig,
        rng: Optional[np.random.Generator] = None,
        batch_size: int = 8,
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
    ) -> MCTSNode[S]:
        num_sims = num_simulations or self.config.num_simulations

        root = MCTSNode(state=root_state)

        root_priors, root_value = self._evaluate_single(root_state)
        self._expand_node_with_priors(root, root_priors)

        if add_dirichlet_noise and root.children:
            self._add_dirichlet_noise(root)

        root.backup(root_value)

        sims_done = 0
        while sims_done < num_sims:
            batch_leaves = []
            batch_paths = []

            batch_count = min(self.batch_size, num_sims - sims_done)
            for _ in range(batch_count):
                leaf, path = self._select_leaf(root)
                if leaf.is_terminal:
                    leaf.backup(leaf.terminal_value)
                else:
                    batch_leaves.append(leaf)
                    batch_paths.append(path)

            if batch_leaves:
                states = [leaf.state for leaf in batch_leaves]
                all_priors, all_values = self.evaluator(states)

                for leaf, priors, value in zip(batch_leaves, all_priors, all_values):
                    self._expand_node_with_priors(leaf, priors)
                    leaf.backup(value)

            sims_done += batch_count

        return root

    def _select_leaf(self, root: MCTSNode[S]) -> Tuple[MCTSNode[S], List[MCTSNode[S]]]:
        """Select a leaf node using UCB, return leaf and path from root."""
        node = root
        path = [node]

        while node.is_expanded and not node.is_terminal:
            node = node.select_child(self.config.c_puct)
            path.append(node)

        return node, path

    def _evaluate_single(self, state: S) -> Tuple[Dict[Action, float], float]:
        """Evaluate a single state (for root)."""
        all_priors, all_values = self.evaluator([state])
        return all_priors[0], all_values[0]

    def _expand_node_with_priors(
        self, node: MCTSNode[S], priors: Dict[Action, float]
    ) -> None:
        """Expand node if not terminal."""
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

        child_states = {
            action: self.game.apply_action(state, action) for action in legal_actions
        }

        node.expand(legal_actions, priors, child_states)

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

    def _add_dirichlet_noise(self, node: MCTSNode[S]) -> None:
        alpha = self.config.dirichlet_alpha
        frac = self.config.dirichlet_frac

        actions = list(node.children.keys())
        noise = self.rng.dirichlet([alpha] * len(actions))

        for action, noise_val in zip(actions, noise):
            child = node.children[action]
            child.prior = (1 - frac) * child.prior + frac * noise_val

    def get_action_probs(
        self,
        root: MCTSNode[S],
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
