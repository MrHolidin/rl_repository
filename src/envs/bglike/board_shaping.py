"""Terminal reward shaping from the final board (per-minion / per-tribe bonuses)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from src.bg_catalog.patch_catalog import race_from_hs_string
from src.bg_core.minion import Minion, Race

from .placement import placement_reward
from .state import BGLikeState


def final_board_for_seat(state: BGLikeState, seat: int) -> Tuple[Minion, ...]:
    """Last recruitment board for ``seat`` at elimination or lobby end."""
    for snap in state.eliminated:
        if snap.seat == seat:
            return tuple(snap.last_board)
    if seat in state.alive:
        return tuple(state.players[seat].board)
    raise ValueError(f"seat {seat} has no final board (lobby not finished for seat)")


def parse_minions_shaping(raw: Any) -> Dict[str, float]:
    """Parse ``minions_shaping: {display_name: bonus}`` from game config."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("minions_shaping must be a mapping minion_name -> bonus")
    out: Dict[str, float] = {}
    for key, value in raw.items():
        name = str(key).strip()
        if not name:
            raise ValueError("minions_shaping keys must be non-empty minion names")
        out[name] = float(value)
    return out


def parse_tribes_shaping(raw: Any) -> Dict[Race, float]:
    """Parse ``tribes_shaping: {TRIBE_NAME: bonus}`` from game config."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("tribes_shaping must be a mapping tribe_name -> bonus")
    out: Dict[Race, float] = {}
    for key, value in raw.items():
        race = race_from_hs_string(str(key).strip().upper())
        if race is Race.ALL:
            raise ValueError("tribes_shaping: Race.ALL is not allowed")
        out[race] = float(value)
    return out


@dataclass(frozen=True)
class BoardShapingConfig:
    minions_shaping: Mapping[str, float]
    tribes_shaping: Mapping[Race, float]

    @classmethod
    def from_params(
        cls,
        *,
        minions_shaping: Any = None,
        tribes_shaping: Any = None,
    ) -> BoardShapingConfig:
        return cls(
            minions_shaping=parse_minions_shaping(minions_shaping),
            tribes_shaping=parse_tribes_shaping(tribes_shaping),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.minions_shaping or self.tribes_shaping)


def minions_shaping_total(
    board: Sequence[Minion],
    bonuses: Mapping[str, float],
) -> float:
    """Sum configured bonus for each board minion matched by ``Minion.name``."""
    if not bonuses:
        return 0.0
    total = 0.0
    for m in board:
        bonus = bonuses.get(m.name)
        if bonus is not None:
            total += float(bonus)
    return total


def tribes_shaping_total(
    board: Sequence[Minion],
    bonuses: Mapping[Race, float],
) -> float:
    """Sum configured bonus for each board minion matched by ``Minion.race``."""
    if not bonuses:
        return 0.0
    total = 0.0
    for m in board:
        if m.race is None or m.race is Race.ALL:
            continue
        bonus = bonuses.get(m.race)
        if bonus is not None:
            total += float(bonus)
    return total


def final_board_shaping_components(
    state: BGLikeState,
    seat: int,
    config: Optional[BoardShapingConfig] = None,
) -> Tuple[float, float, float]:
    """Return ``(minions_term, tribes_term, total)`` shaping for ``seat``."""
    cfg = config or BoardShapingConfig({}, {})
    board = final_board_for_seat(state, seat)
    minion_term = minions_shaping_total(board, cfg.minions_shaping)
    tribe_term = tribes_shaping_total(board, cfg.tribes_shaping)
    return minion_term, tribe_term, minion_term + tribe_term


def final_board_shaping(
    state: BGLikeState,
    seat: int,
    config: Optional[BoardShapingConfig] = None,
) -> float:
    _, _, total = final_board_shaping_components(state, seat, config)
    return total


def terminal_reward_for_seat(
    state: BGLikeState,
    seat: int,
    place: int,
    config: Optional[BoardShapingConfig] = None,
) -> float:
    """Placement reward plus optional final-board shaping."""
    return placement_reward(place) + final_board_shaping(state, seat, config)


def terminal_reward_breakdown(
    state: BGLikeState,
    seat: int,
    place: int,
    config: Optional[BoardShapingConfig] = None,
) -> Mapping[str, float]:
    """Diagnostic breakdown for replays / info dicts."""
    minion_term, tribe_term, shaping = final_board_shaping_components(
        state,
        seat,
        config,
    )
    return {
        "placement_reward_base": placement_reward(place),
        "minions_shaping": minion_term,
        "tribes_shaping": tribe_term,
        "board_shaping_total": shaping,
        "placement_reward": placement_reward(place) + shaping,
    }


__all__ = [
    "BoardShapingConfig",
    "final_board_for_seat",
    "final_board_shaping",
    "final_board_shaping_components",
    "minions_shaping_total",
    "parse_minions_shaping",
    "parse_tribes_shaping",
    "terminal_reward_breakdown",
    "terminal_reward_for_seat",
    "tribes_shaping_total",
]
