"""MCTS node with lazy child state creation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Generic, List, Optional, TypeVar, Callable

from src.games.turn_based_game import Action, TurnBasedGame

S = TypeVar("S")


@dataclass
class LazyMCTSNode(Generic[S]):
    """
    MCTS node with lazy child state creation.
    
    Child states are only created when the child is first selected,
    not when the node is expanded. This saves ~7x apply_action calls.
    """

    state: S
    parent: Optional["LazyMCTSNode[S]"] = None
    action_from_parent: Optional[Action] = None

    visit_count: int = 0
    total_value: float = 0.0
    prior: float = 0.0

    children: Dict[Action, "LazyMCTSNode[S]"] = field(default_factory=dict)
    _pending_actions: Optional[List[Action]] = None
    _priors: Optional[Dict[Action, float]] = None
    _game: Optional[TurnBasedGame[S]] = None
    
    is_expanded: bool = False
    is_terminal: bool = False
    terminal_value: Optional[float] = None

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb_score(self, c_puct: float, parent_visits: int) -> float:
        # Q is from child's perspective; parent wants -Q (opponent's loss = parent's gain)
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return -self.q_value + exploration

    def select_child(self, c_puct: float) -> "LazyMCTSNode[S]":
        """Select child with highest UCB, creating state lazily if needed."""
        if not self.children and not self._pending_actions:
            raise ValueError("Node has no children")

        best_score = -math.inf
        best_action = None
        best_is_pending = False

        for action, child in self.children.items():
            score = child.ucb_score(c_puct, self.visit_count)
            if score > best_score:
                best_score = score
                best_action = action
                best_is_pending = False

        if self._pending_actions:
            for action in self._pending_actions:
                prior = self._priors.get(action, 0.0)
                score = c_puct * prior * math.sqrt(self.visit_count)
                if score > best_score:
                    best_score = score
                    best_action = action
                    best_is_pending = True

        if best_is_pending:
            child_state = self._game.apply_action(self.state, best_action)
            child = LazyMCTSNode(
                state=child_state,
                parent=self,
                action_from_parent=best_action,
                prior=self._priors.get(best_action, 0.0),
            )
            self.children[best_action] = child
            self._pending_actions.remove(best_action)
            return child

        return self.children[best_action]

    def expand_lazy(
        self,
        legal_actions: List[Action],
        priors: Dict[Action, float],
        game: TurnBasedGame[S],
    ) -> None:
        """Mark node as expanded with pending actions (no states created yet)."""
        self._pending_actions = list(legal_actions)
        self._priors = priors
        self._game = game
        self.is_expanded = True

    def backup(self, value: float) -> None:
        node = self
        current_value = value

        while node is not None:
            node.visit_count += 1
            node.total_value += current_value
            current_value = -current_value
            node = node.parent

    def get_policy_distribution(self, temperature: float) -> Dict[Action, float]:
        if not self.children:
            return {}

        if temperature == 0.0:
            best_action = max(self.children.keys(), key=lambda a: self.children[a].visit_count)
            return {a: (1.0 if a == best_action else 0.0) for a in self.children}

        visits = {a: child.visit_count for a, child in self.children.items()}

        if temperature == 1.0:
            total = sum(visits.values())
            if total == 0:
                n = len(visits)
                return {a: 1.0 / n for a in visits}
            return {a: v / total for a, v in visits.items()}

        exp_visits = {a: v ** (1.0 / temperature) for a, v in visits.items()}
        total = sum(exp_visits.values())
        if total == 0:
            n = len(exp_visits)
            return {a: 1.0 / n for a in exp_visits}
        return {a: v / total for a, v in exp_visits.items()}
