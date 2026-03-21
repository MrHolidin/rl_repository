"""MCTS tree node."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Generic, List, Optional, TypeVar

from src.games.turn_based_game import Action

S = TypeVar("S")


@dataclass
class MCTSNode(Generic[S]):
    """A node in the MCTS tree."""

    state: S
    parent: Optional["MCTSNode[S]"] = None
    action_from_parent: Optional[Action] = None

    visit_count: int = 0
    total_value: float = 0.0
    prior: float = 0.0

    children: Dict[Action, "MCTSNode[S]"] = field(default_factory=dict)
    is_expanded: bool = False
    is_terminal: bool = False
    terminal_value: Optional[float] = None

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb_score(self, c_puct: float, parent_visits: int) -> float:
        # Q is stored from this node's (child's) perspective
        # Parent wants to maximize their outcome = minimize child's outcome
        # So we use -q_value (negate child's Q to get parent's Q)
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return -self.q_value + exploration

    def select_child(self, c_puct: float) -> "MCTSNode[S]":
        if not self.children:
            raise ValueError("Node has no children")

        best_score = -math.inf
        best_child = None

        for child in self.children.values():
            score = child.ucb_score(c_puct, self.visit_count)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def expand(
        self,
        legal_actions: List[Action],
        priors: Dict[Action, float],
        child_states: Dict[Action, S],
    ) -> None:
        for action in legal_actions:
            self.children[action] = MCTSNode(
                state=child_states[action],
                parent=self,
                action_from_parent=action,
                prior=priors.get(action, 1.0 / len(legal_actions)),
            )
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
