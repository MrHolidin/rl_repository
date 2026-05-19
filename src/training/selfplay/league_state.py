"""Host-side league state: slot stats, eviction, snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .league_policy import selection_probabilities

SLOT_CURRENT: int = -1
SLOT_SCRIPTED: int = -2


@dataclass
class LeagueSlot:
    slot_id: int
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    ema_win_rate: float = 0.5
    checkpoint_path: Optional[str] = None
    episode: Optional[int] = None
    weights_bytes: Optional[bytes] = None

    @property
    def cumulative_win_rate(self) -> float:
        if self.games <= 0:
            return 0.5
        return self.wins / self.games


@dataclass
class LeagueSnapshot:
    """Immutable view broadcast to workers (or used locally for sampling)."""

    win_rates: Dict[int, float]
    frozen_slot_ids: List[int]
    epoch: int = 0


class LeagueController:
    """Central EMA win-rate tracker and frozen-slot registry."""

    def __init__(self, *, ema_beta: float = 0.05) -> None:
        self._beta = ema_beta
        self._slots: Dict[int, LeagueSlot] = {}
        self._next_frozen_id = 0
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def register_meta_slot(self, slot_id: int) -> None:
        self._slots[slot_id] = LeagueSlot(slot_id=slot_id)

    def frozen_slots(self) -> List[LeagueSlot]:
        return [s for s in self._slots.values() if s.slot_id >= 0]

    def get_slot(self, slot_id: int) -> Optional[LeagueSlot]:
        return self._slots.get(slot_id)

    def add_frozen_checkpoint(self, checkpoint_path: str, episode: int) -> int:
        slot_id = self._next_frozen_id
        self._next_frozen_id += 1
        self._slots[slot_id] = LeagueSlot(
            slot_id=slot_id,
            checkpoint_path=checkpoint_path,
            episode=episode,
        )
        return slot_id

    def add_frozen_bytes(self, weights_bytes: bytes, *, episode: Optional[int] = None) -> int:
        slot_id = self._next_frozen_id
        self._next_frozen_id += 1
        self._slots[slot_id] = LeagueSlot(
            slot_id=slot_id,
            weights_bytes=weights_bytes,
            episode=episode if episode is not None else slot_id,
        )
        return slot_id

    def remove_slot(self, slot_id: int) -> None:
        self._slots.pop(slot_id, None)

    def apply_outcomes(self, outcomes: List[Tuple[int, int]]) -> None:
        for slot_id, agent_result in outcomes:
            stats = self._slots.get(slot_id)
            if stats is None:
                continue
            stats.games += 1
            if agent_result == 1:
                stats.losses += 1
                y = 0.0
            elif agent_result == -1:
                stats.wins += 1
                y = 1.0
            else:
                stats.draws += 1
                y = 0.5
            stats.ema_win_rate = (1.0 - self._beta) * stats.ema_win_rate + self._beta * y
        if outcomes:
            self._epoch += 1

    def win_rates(self) -> Dict[int, float]:
        return {sid: s.ema_win_rate for sid, s in self._slots.items() if sid >= 0}

    def snapshot(self) -> LeagueSnapshot:
        frozen = sorted(s.slot_id for s in self.frozen_slots())
        return LeagueSnapshot(win_rates=self.win_rates(), frozen_slot_ids=frozen, epoch=self._epoch)

    def evict_worst_ema(self, max_size: int) -> None:
        if max_size <= 0:
            for sid in list(self.win_rates().keys()):
                self.remove_slot(sid)
            return
        frozen = self.frozen_slots()
        while len(frozen) > max_size:
            worst = min(frozen, key=lambda s: (s.ema_win_rate, s.episode if s.episode is not None else s.slot_id))
            self.remove_slot(worst.slot_id)
            frozen = self.frozen_slots()

    def get_frozen_stats_for_status(self, *, pfsp_eps: float = 1e-2) -> List[Dict[str, Any]]:
        frozen = sorted(self.frozen_slots(), key=lambda s: s.slot_id)
        if not frozen:
            return []
        ema_rates = [s.ema_win_rate for s in frozen]
        probs = selection_probabilities(ema_rates, eps=pfsp_eps)
        out: List[Dict[str, Any]] = []
        for slot, p in zip(frozen, probs):
            row: Dict[str, Any] = {
                "slot_id": slot.slot_id,
                "games": slot.games,
                "wins": slot.wins,
                "losses": slot.losses,
                "draws": slot.draws,
                "ema_win_rate": round(slot.ema_win_rate, 4),
                "cumulative_win_rate": round(slot.cumulative_win_rate, 4),
                "win_rate": round(slot.ema_win_rate, 4),
                "selection_probability": round(p, 4),
            }
            if slot.checkpoint_path is not None:
                import os

                row["checkpoint"] = os.path.basename(slot.checkpoint_path)
            if slot.episode is not None:
                row["episode"] = slot.episode
            out.append(row)
        return out


__all__ = [
    "LeagueController",
    "LeagueSlot",
    "LeagueSnapshot",
    "SLOT_CURRENT",
    "SLOT_SCRIPTED",
]
