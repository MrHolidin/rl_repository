"""Per-seat controller configuration for an 8-player lobby."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

from src.agents.base_agent import BaseAgent
from src.agents.random_agent import RandomAgent

from .actions import NUM_PLAYERS


class SeatKind(str, Enum):
    RANDOM = "random"
    LEARNED = "learned"


@dataclass
class SeatConfig:
    kind: SeatKind
    agent: Optional[BaseAgent] = None

    def __post_init__(self) -> None:
        if self.kind == SeatKind.LEARNED and self.agent is None:
            raise ValueError("LEARNED seat requires agent")


def default_random_lobby() -> Tuple[SeatConfig, ...]:
    return tuple(SeatConfig(SeatKind.RANDOM) for _ in range(NUM_PLAYERS))


def build_training_lobby_configs(
    current_seats: Sequence[int],
    current_agent: BaseAgent,
    opponents_by_seat: Dict[int, BaseAgent],
    *,
    seed: Optional[int] = None,
) -> Tuple[SeatConfig, ...]:
    """``current_seats`` share ``current_agent``; each other seat uses its own opponent agent."""
    del seed  # reserved for API stability
    current = set(current_seats)
    for s in current_seats:
        if not 0 <= s < NUM_PLAYERS:
            raise ValueError(f"invalid seat {s}")
    configs: List[SeatConfig] = []
    for seat in range(NUM_PLAYERS):
        if seat in current:
            configs.append(SeatConfig(SeatKind.LEARNED, current_agent))
        else:
            opp = opponents_by_seat.get(seat)
            if opp is None:
                raise ValueError(f"opponent seat {seat} missing from opponents_by_seat")
            configs.append(SeatConfig(SeatKind.LEARNED, opp))
    return tuple(configs)


def lobby_from_learned_seats(
    learned_seats: Sequence[int],
    *,
    agent_by_seat: Optional[Dict[int, BaseAgent]] = None,
    seed: Optional[int] = None,
) -> Tuple[SeatConfig, ...]:
    """Build 8 seat configs: learned seats use agents, others random."""
    if not learned_seats:
        raise ValueError("learned_seats must be non-empty")
    for s in learned_seats:
        if not 0 <= s < NUM_PLAYERS:
            raise ValueError(f"invalid seat {s}")
    agent_by_seat = agent_by_seat or {}
    configs: list[SeatConfig] = []
    rng_seed = seed
    for seat in range(NUM_PLAYERS):
        if seat in learned_seats:
            agent = agent_by_seat.get(seat)
            if agent is None:
                raise ValueError(f"learned seat {seat} missing agent in agent_by_seat")
            configs.append(SeatConfig(SeatKind.LEARNED, agent))
        else:
            rs = (rng_seed + 1000 + seat) if rng_seed is not None else None
            configs.append(SeatConfig(SeatKind.RANDOM, RandomAgent(seed=rs)))
    return tuple(configs)


__all__ = [
    "SeatConfig",
    "SeatKind",
    "build_training_lobby_configs",
    "default_random_lobby",
    "lobby_from_learned_seats",
]
