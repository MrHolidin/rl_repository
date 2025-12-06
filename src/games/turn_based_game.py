from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Sequence, TypeVar, Optional

S = TypeVar("S")  # тип состояния
Action = int  # пока считаем, что действия — целые индексы


class TurnBasedGame(ABC, Generic[S]):
    """
    Общий интерфейс для детерминированной двухигровой игры с совершенной информацией.
    Никаких env, только "чистые" правила.
    """

    @abstractmethod
    def legal_actions(self, state: S) -> Sequence[Action]:
        """Все допустимые действия в данном состоянии."""

    @abstractmethod
    def apply_action(self, state: S, action: Action) -> S:
        """Вернуть новое состояние после хода."""

    @abstractmethod
    def current_player(self, state: S) -> int:
        """
        Какой игрок ходит сейчас:
        обычно 1 для "макс" и -1 для "мин".
        """

    @abstractmethod
    def is_terminal(self, state: S) -> bool:
        """Конечное ли состояние (победа/ничья/поражение)?"""

    @abstractmethod
    def winner(self, state: S) -> Optional[int]:
        """
        Кто победил:

        * 1  — выиграл игрок с токеном +1
        * -1 — выиграл игрок с токеном -1
        * 0  — ничья
        * None — ещё не закончено
        """

