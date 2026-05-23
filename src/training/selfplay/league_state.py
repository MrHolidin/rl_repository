"""Host-side league controller: registry + rating glue."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

AgentOutcome = Union[int, float]

from .game_record import (
    GameRecord,
    SLOT_CURRENT,
    SLOT_SCRIPTED,
)
from .league_policy import selection_probabilities
from .placement_ema import PlacementEmaTracker
from .rating_system import EmaPairwiseRating, RatingSystem, TrueSkillRating, make_rating_system
from .slot_registry import SlotRegistry


def normalize_agent_score(agent_result: AgentOutcome) -> float:
    """Map legacy ±1/0 or a [0, 1] score to learner performance in [0, 1]."""
    if isinstance(agent_result, float):
        return max(0.0, min(1.0, agent_result))
    if agent_result == 1:
        return 1.0
    if agent_result == -1:
        return 0.0
    return 0.5


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
    rating_kind: str = "ema"
    trueskill: Dict[int, Tuple[float, float]] = field(default_factory=dict)


class LeagueController:
    """Thin glue: ``SlotRegistry`` + ``RatingSystem``."""

    def __init__(
        self,
        *,
        registry: Optional[SlotRegistry] = None,
        rating: Optional[RatingSystem] = None,
        ema_beta: float = 0.05,
        rating_kind: str = "ema",
        trueskill: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._registry = registry or SlotRegistry()
        self._rating = rating or make_rating_system(
            rating_kind,
            ema_beta=ema_beta,
            trueskill=trueskill,
        )
        self._slots: Dict[int, LeagueSlot] = {}
        self._epoch = 0
        self._slot_id_to_scripted_key: Dict[int, str] = {}
        self._rating_kind = rating_kind
        self._placement_ema = PlacementEmaTracker(window=20)

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def registry(self) -> SlotRegistry:
        return self._registry

    @property
    def rating(self) -> RatingSystem:
        return self._rating

    @property
    def rating_kind(self) -> str:
        return self._rating_kind

    @property
    def slot_id_to_scripted_key(self) -> Dict[int, str]:
        return dict(self._slot_id_to_scripted_key)

    def register_scripted_slots(self, scripted_slot_ids: Dict[str, int]) -> None:
        """Register one meta slot per scripted bot key."""
        self._slot_id_to_scripted_key = {
            int(sid): str(key) for key, sid in scripted_slot_ids.items()
        }
        for sid in self._slot_id_to_scripted_key:
            self.register_meta_slot(int(sid))

    def pool_slot_ids(self) -> List[int]:
        """Non-current pool ids: per-key scripted meta + frozen checkpoints."""
        scripted = sorted(self._slot_id_to_scripted_key.keys())
        frozen = self._registry.frozen_ids()
        return scripted + frozen

    def register_meta_slot(self, slot_id: int) -> None:
        self._registry.register_meta_slot(slot_id)
        self._rating.register(slot_id)
        self._sync_view(slot_id)

    def frozen_slots(self) -> List[LeagueSlot]:
        return [self._sync_view(sid) for sid in self._registry.frozen_ids()]

    def get_slot(self, slot_id: int) -> Optional[LeagueSlot]:
        if self._registry.get(slot_id) is None:
            return None
        return self._sync_view(slot_id)

    def add_frozen_checkpoint(self, checkpoint_path: str, episode: int) -> int:
        slot_id = self._registry.add(checkpoint_path=checkpoint_path, episode=episode)
        self._rating.register(slot_id)
        self._sync_view(slot_id)
        return slot_id

    def add_frozen_bytes(self, weights_bytes: bytes, *, episode: Optional[int] = None) -> int:
        slot_id = self._registry.add_bytes(weights_bytes, episode=episode)
        self._rating.register(slot_id)
        self._sync_view(slot_id)
        return slot_id

    def remove_slot(self, slot_id: int) -> None:
        self._registry.remove(slot_id)
        self._rating.remove(slot_id)
        self._placement_ema.remove(slot_id)
        self._slots.pop(int(slot_id), None)

    def submit(self, record: GameRecord) -> None:
        if self._rating.update(record):
            self._epoch += 1
            for p in record.participants:
                self._placement_ema.record(p.slot_id, p.placement)
                if self._registry.get(p.slot_id) is not None:
                    self._sync_view(p.slot_id)

    def apply_outcomes(self, records: Sequence[GameRecord]) -> None:
        for record in records:
            self.submit(record)

    def set_ema_win_rate(self, slot_id: int, value: float) -> None:
        """Direct EMA override (tests / manual tuning)."""
        if isinstance(self._rating, EmaPairwiseRating):
            stats = self._rating.get_stats(int(slot_id))
            if stats is not None:
                stats.ema_win_rate = float(value)
        slot = self._slots.get(int(slot_id))
        if slot is not None:
            slot.ema_win_rate = float(value)

    def win_rates(self) -> Dict[int, float]:
        return {sid: self._rating.rating(sid) for sid in self._registry.frozen_ids()}

    def pool_win_rates(self) -> Dict[int, float]:
        """Win-rate / strength scalars for PFSP over scripted + frozen + current."""
        ids = self.pool_slot_ids() + [SLOT_CURRENT]
        return {sid: self._rating.rating(sid) for sid in ids}

    def sync_snapshot(self) -> LeagueSnapshot:
        pool_ids = self.pool_slot_ids()
        trueskill: Dict[int, Tuple[float, float]] = {}
        if isinstance(self._rating, TrueSkillRating):
            for sid in pool_ids + [SLOT_CURRENT]:
                trueskill[int(sid)] = self._rating.get_mu_sigma(int(sid))
        return LeagueSnapshot(
            win_rates=self.pool_win_rates(),
            frozen_slot_ids=list(self._registry.frozen_ids()),
            epoch=self._epoch,
            rating_kind=self._rating_kind,
            trueskill=trueskill,
        )

    def snapshot(self) -> LeagueSnapshot:
        return self.sync_snapshot()

    def evict_worst(self, max_size: int) -> None:
        if max_size <= 0:
            for sid in list(self._registry.frozen_ids()):
                self.remove_slot(sid)
            return
        frozen = self._registry.frozen_entries()
        while len(frozen) > max_size:
            worst = min(
                frozen,
                key=lambda e: self._rating.eviction_sort_key(
                    e.slot_id,
                    episode=e.episode if e.episode is not None else e.slot_id,
                ),
            )
            self.remove_slot(worst.slot_id)
            frozen = self._registry.frozen_entries()

    def get_status_file_data(self, *, pfsp_eps: float = 1e-2) -> Dict[str, Any]:
        rows = self.get_pool_stats_for_status(pfsp_eps=pfsp_eps)
        if self._uses_trueskill_status():
            return {"rating_system": "trueskill", "agents": rows}
        return {"frozen_agents": rows}

    def get_pool_stats_for_status(self, *, pfsp_eps: float = 1e-2) -> List[Dict[str, Any]]:
        if self._uses_trueskill_status():
            return self._trueskill_pool_stats_for_status()
        return self._ema_pool_stats_for_status(pfsp_eps=pfsp_eps)

    def _uses_trueskill_status(self) -> bool:
        return isinstance(self._rating, TrueSkillRating) or (
            self._rating_kind.strip().lower() == "trueskill"
        )

    def _placement_status_fields(self, slot_id: int) -> Dict[str, Any]:
        ema = self._placement_ema.get(slot_id)
        if ema is None:
            return {}
        return {"placement_ema": round(float(ema), 4)}

    def _iter_pool_status_targets(
        self,
    ) -> Iterator[Tuple[int, str, Optional[str], Optional[LeagueSlot]]]:
        """Yield ``(slot_id, kind, label, slot)`` in current → scripted → frozen order."""
        if self._current_learner_registered():
            yield SLOT_CURRENT, "current", "learner", None
        for sid in sorted(self._slot_id_to_scripted_key.keys()):
            yield sid, "scripted", self._slot_id_to_scripted_key[sid], None
        for slot in sorted(self.frozen_slots(), key=lambda s: s.slot_id):
            yield slot.slot_id, "frozen", None, slot

    def _status_row(
        self,
        slot_id: int,
        *,
        kind: str,
        label: Optional[str] = None,
        slot: Optional[LeagueSlot] = None,
    ) -> Dict[str, Any]:
        sm = self._rating.summary(slot_id)
        row: Dict[str, Any] = {
            "slot_id": slot_id,
            "kind": kind,
            "games": sm["games"],
        }
        if label is not None:
            row["label"] = label
        row.update(self._placement_status_fields(slot_id))
        if slot is not None:
            if slot.checkpoint_path is not None:
                row["checkpoint"] = os.path.basename(slot.checkpoint_path)
            if slot.episode is not None:
                row["episode"] = slot.episode
        return row

    def _trueskill_status_row(
        self,
        slot_id: int,
        *,
        kind: str,
        label: Optional[str] = None,
        slot: Optional[LeagueSlot] = None,
    ) -> Dict[str, Any]:
        row = self._status_row(slot_id, kind=kind, label=label, slot=slot)
        sm = self._rating.summary(slot_id)
        row["mu"] = sm["mu"]
        row["sigma"] = sm["sigma"]
        if isinstance(self._rating, TrueSkillRating) and kind != "current":
            row["match_quality_vs_current"] = round(
                self._rating.match_quality_vs_current(slot_id), 4
            )
        return row

    def _ema_status_row(
        self,
        slot_id: int,
        *,
        kind: str,
        label: Optional[str] = None,
        slot: Optional[LeagueSlot] = None,
        selection_probability: Optional[float] = None,
    ) -> Dict[str, Any]:
        row = self._status_row(slot_id, kind=kind, label=label, slot=slot)
        sm = self._rating.summary(slot_id)
        row.update(
            {
                "wins": sm["wins"],
                "losses": sm["losses"],
                "draws": sm["draws"],
                "ema_win_rate": sm["ema_win_rate"],
                "cumulative_win_rate": sm["cumulative_win_rate"],
                "win_rate": sm["win_rate"],
            }
        )
        if selection_probability is not None:
            row["selection_probability"] = round(selection_probability, 4)
        return row

    def _current_learner_registered(self) -> bool:
        return self._registry.get(SLOT_CURRENT) is not None

    def _trueskill_pool_stats_for_status(self) -> List[Dict[str, Any]]:
        return [
            self._trueskill_status_row(slot_id, kind=kind, label=label, slot=slot)
            for slot_id, kind, label, slot in self._iter_pool_status_targets()
        ]

    def _ema_pool_stats_for_status(self, *, pfsp_eps: float = 1e-2) -> List[Dict[str, Any]]:
        targets = list(self._iter_pool_status_targets())
        frozen_probs: Dict[int, float] = {}
        frozen_targets = [(sid, slot) for sid, kind, _, slot in targets if kind == "frozen"]
        if frozen_targets:
            frozen_ids = [sid for sid, _ in frozen_targets]
            rates = [self._rating.rating(sid) for sid in frozen_ids]
            probs = selection_probabilities(rates, eps=pfsp_eps)
            frozen_probs = dict(zip(frozen_ids, probs))
        return [
            self._ema_status_row(
                slot_id,
                kind=kind,
                label=label,
                slot=slot,
                selection_probability=frozen_probs.get(slot_id),
            )
            for slot_id, kind, label, slot in targets
        ]

    def _sync_view(self, slot_id: int) -> LeagueSlot:
        sid = int(slot_id)
        entry = self._registry.get(sid)
        if entry is None:
            raise KeyError(f"missing registry slot {sid}")
        sm = self._rating.summary(sid)
        slot = self._slots.get(sid)
        if slot is None:
            slot = LeagueSlot(slot_id=sid)
            self._slots[sid] = slot
        slot.games = int(sm["games"])
        slot.wins = int(sm["wins"])
        slot.losses = int(sm["losses"])
        slot.draws = int(sm["draws"])
        slot.ema_win_rate = float(sm["ema_win_rate"])
        slot.checkpoint_path = entry.checkpoint_path
        slot.episode = entry.episode
        slot.weights_bytes = entry.weights_bytes
        return slot


__all__ = [
    "AgentOutcome",
    "GameRecord",
    "LeagueController",
    "LeagueSlot",
    "LeagueSnapshot",
    "SLOT_CURRENT",
    "SLOT_SCRIPTED",
    "normalize_agent_score",
]
