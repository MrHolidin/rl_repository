"""Universal game outcome records for league rating updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from src.envs.bglike.placement import pairwise_learner_score

SLOT_CURRENT: int = -1
SLOT_SCRIPTED: int = -2


def build_scripted_slot_map(
    scripted_keys: Sequence[str],
    *,
    start: int = SLOT_SCRIPTED,
) -> Dict[str, int]:
    """Map each scripted bot key to a unique negative meta slot id."""
    out: Dict[str, int] = {}
    next_id = int(start)
    for key in sorted(scripted_keys):
        out[str(key)] = next_id
        next_id -= 1
    return out


def invert_scripted_slot_map(scripted_slot_ids: Dict[str, int]) -> Dict[int, str]:
    return {sid: key for key, sid in scripted_slot_ids.items()}


LeagueUpdate = Tuple[int, float]


@dataclass(frozen=True)
class ParticipantOutcome:
    slot_id: int
    placement: int
    scripted_key: Optional[str] = None


@dataclass(frozen=True)
class GameRecord:
    participants: Tuple[ParticipantOutcome, ...]


def league_updates_from_record(record: GameRecord) -> List[LeagueUpdate]:
    """Pairwise learner-vs-opponent scores for EMA updates."""
    learners = [p for p in record.participants if p.slot_id == SLOT_CURRENT]
    opponents = [p for p in record.participants if p.slot_id != SLOT_CURRENT]
    if not learners or not opponents:
        return []
    out: List[LeagueUpdate] = []
    for learner in learners:
        for opp in opponents:
            score = pairwise_learner_score(learner.placement, opp.placement)
            out.append((opp.slot_id, score))
    return out


def minibg_record_from_learner_score(
    slot_id: int,
    learner_score: float,
    *,
    scripted_key: Optional[str] = None,
) -> GameRecord:
    """Two-player record from normalized learner performance in [0, 1]."""
    if learner_score > 0.5:
        learner_place, opp_place = 1, 2
    elif learner_score < 0.5:
        learner_place, opp_place = 2, 1
    else:
        learner_place, opp_place = 1, 1
    return GameRecord(
        participants=(
            ParticipantOutcome(SLOT_CURRENT, learner_place),
            ParticipantOutcome(int(slot_id), opp_place, scripted_key=scripted_key),
        )
    )


def game_record_for_lobby_end(
    *,
    current_seats: Sequence[int],
    slot_by_seat: Dict[int, int],
    placements_full: Dict[int, int],
    slot_id_to_scripted_key: Optional[Dict[int, str]] = None,
) -> Optional[GameRecord]:
    """One full-lobby ``GameRecord`` when all placements are known."""
    if not placements_full:
        return None
    key_map = slot_id_to_scripted_key or {}
    current = {int(s) for s in current_seats}
    participants: List[ParticipantOutcome] = []
    for seat, place in sorted(placements_full.items()):
        sid = int(seat)
        p = int(place)
        if sid in current:
            participants.append(ParticipantOutcome(SLOT_CURRENT, p))
        elif sid in slot_by_seat:
            slot_id = int(slot_by_seat[sid])
            participants.append(
                ParticipantOutcome(
                    slot_id,
                    p,
                    scripted_key=key_map.get(slot_id),
                )
            )
    if len(participants) < 2:
        return None
    return GameRecord(participants=tuple(participants))


__all__ = [
    "GameRecord",
    "LeagueUpdate",
    "ParticipantOutcome",
    "SLOT_CURRENT",
    "SLOT_SCRIPTED",
    "build_scripted_slot_map",
    "game_record_for_lobby_end",
    "invert_scripted_slot_map",
    "league_updates_from_record",
    "minibg_record_from_learner_score",
]
