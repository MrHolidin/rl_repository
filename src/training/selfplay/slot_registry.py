"""Frozen opponent pool registry (IDs and weights, no rating stats)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SlotEntry:
    slot_id: int
    checkpoint_path: Optional[str] = None
    episode: Optional[int] = None
    weights_bytes: Optional[bytes] = None


class SlotRegistry:
    """Who is in the pool: slot IDs, checkpoints, and weight bytes."""

    def __init__(self) -> None:
        self._entries: Dict[int, SlotEntry] = {}
        self._next_frozen_id = 0

    def register_meta_slot(self, slot_id: int) -> None:
        self._entries[int(slot_id)] = SlotEntry(slot_id=int(slot_id))

    def add(self, *, checkpoint_path: str, episode: int) -> int:
        slot_id = self._next_frozen_id
        self._next_frozen_id += 1
        self._entries[slot_id] = SlotEntry(
            slot_id=slot_id,
            checkpoint_path=checkpoint_path,
            episode=episode,
        )
        return slot_id

    def add_bytes(self, weights_bytes: bytes, *, episode: Optional[int] = None) -> int:
        slot_id = self._next_frozen_id
        self._next_frozen_id += 1
        self._entries[slot_id] = SlotEntry(
            slot_id=slot_id,
            weights_bytes=weights_bytes,
            episode=episode if episode is not None else slot_id,
        )
        return slot_id

    def remove(self, slot_id: int) -> None:
        self._entries.pop(int(slot_id), None)

    def get(self, slot_id: int) -> Optional[SlotEntry]:
        return self._entries.get(int(slot_id))

    def all_ids(self) -> List[int]:
        return sorted(self._entries.keys())

    def frozen_ids(self) -> List[int]:
        return sorted(sid for sid in self._entries if sid >= 0)

    def frozen_entries(self) -> List[SlotEntry]:
        return [self._entries[sid] for sid in self.frozen_ids()]


__all__ = ["SlotEntry", "SlotRegistry"]
