"""Core MCTS algorithm."""

from __future__ import annotations

from typing import Callable, Dict, Generic, List, Optional, Tuple, TypeVar

import numpy as np

from src.games.turn_based_game import Action, TurnBasedGame

from .config import MCTSConfig
from .node import MCTSNode

S = TypeVar("S")

NetworkEvaluator = Callable[[S], Tuple[Dict[Action, float], float]]


class MCTS(Generic[S]):
    """Monte Carlo Tree Search with neural network evaluation."""

    def __init__(
        self,
        game: TurnBasedGame[S],
        evaluator: NetworkEvaluator,
        config: MCTSConfig,
        rng: Optional[np.random.Generator] = None,
    ):
        self.game = game
        self.evaluator = evaluator
        self.config = config
        self.rng = rng or np.random.default_rng()

    def search(
        self,
        root_state: S,
        num_simulations: Optional[int] = None,
        add_dirichlet_noise: bool = True,
    ) -> MCTSNode[S]:
        num_sims = num_simulations or self.config.num_simulations

        root = MCTSNode(state=root_state)
        self._expand_node(root)

        if add_dirichlet_noise and root.children:
            self._add_dirichlet_noise(root)

        for _ in range(num_sims):
            self._simulate(root)

        return root

    def _simulate(self, root: MCTSNode[S]) -> None:
        node = root

        while node.is_expanded and not node.is_terminal:
            node = node.select_child(self.config.c_puct)

        if node.is_terminal:
            value = node.terminal_value
            node.backup(value)
            return

        value = self._expand_node(node)
        node.backup(value)

    def _expand_node(self, node: MCTSNode[S]) -> float:
        state = node.state

        if self.game.is_terminal(state):
            node.is_terminal = True
            winner = self.game.winner(state)
            current_player = self.game.current_player(state)

            if winner is None or winner == 0:
                value = 0.0
            elif winner == current_player:
                value = 1.0
            else:
                value = -1.0

            node.terminal_value = value
            return value

        legal_actions = list(self.game.legal_actions(state))
        if not legal_actions:
            node.is_terminal = True
            node.terminal_value = 0.0
            return 0.0

        priors, value = self.evaluator(state)
        priors = self._normalize_priors(priors, legal_actions)

        child_states = {
            action: self.game.apply_action(state, action)
            for action in legal_actions
        }

        node.expand(legal_actions, priors, child_states)
        return value

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
