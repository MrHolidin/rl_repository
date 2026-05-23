"""Rolling placement EMA for BGLike league slots (lower place = better)."""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional


class PlacementEmaTracker:
    """EMA of finish placement over the last ``window`` lobby games per slot."""

    def __init__(self, *, window: int = 20) -> None:
        self._window = max(1, int(window))
        self._alpha = 2.0 / (self._window + 1)
        self._history: Dict[int, Deque[float]] = {}
        self._ema: Dict[int, float] = {}

    @property
    def window(self) -> int:
        return self._window

    def record(self, slot_id: int, placement: int) -> None:
        sid = int(slot_id)
        place = float(placement)
        hist = self._history.setdefault(sid, deque(maxlen=self._window))
        hist.append(place)
        prev = self._ema.get(sid)
        if prev is None:
            self._ema[sid] = place
        else:
            self._ema[sid] = self._alpha * place + (1.0 - self._alpha) * prev

    def get(self, slot_id: int) -> Optional[float]:
        return self._ema.get(int(slot_id))

    def games_in_window(self, slot_id: int) -> int:
        hist = self._history.get(int(slot_id))
        return len(hist) if hist is not None else 0

    def remove(self, slot_id: int) -> None:
        sid = int(slot_id)
        self._history.pop(sid, None)
        self._ema.pop(sid, None)


__all__ = ["PlacementEmaTracker"]
