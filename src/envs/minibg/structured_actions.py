"""Structured MiniBG actions for policies that score legal tuples (type + args)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import FrozenSet, Sequence, Tuple

from .actions import BOARD_SIZE, HAND_SIZE, SHOP_SIZE


class StructActionType(IntEnum):
    ROLL = 0
    LEVEL_UP = 1
    BUY = 2
    SELL = 3
    PLACE = 4
    COMPLETE_TURN = 5


@dataclass(frozen=True)
class StructAction:
    """Frozen action token; COMPLETE_TURN carries board order via ``step(..., board_perm=...)``."""

    type: StructActionType
    args: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        validate_struct_action(self)


def validate_board_perm(perm: Tuple[int, ...]) -> None:
    if len(perm) != BOARD_SIZE:
        raise ValueError(f"board_perm must have length {BOARD_SIZE}, got {len(perm)}")
    if tuple(sorted(perm)) != tuple(range(BOARD_SIZE)):
        raise ValueError(f"board_perm must be a permutation of 0..{BOARD_SIZE - 1}, got {perm}")


def slot_pick_sequence_to_perm(picked_slots: Sequence[int], num_board_minions: int) -> Tuple[int, ...]:
    """Turn autoregressive picks (occupied slot indices) into a full ``BOARD_SIZE`` permutation for ``reorder_board``.

    Board minions occupy contiguous list indices ``0 .. k-1`` before reorder; ``picked_slots`` lists which
    old slot to place next (length ``k``, unique).
    """
    k = int(num_board_minions)
    if k < 0 or k > BOARD_SIZE:
        raise ValueError(f"num_board_minions must be in [0, {BOARD_SIZE}], got {k}")
    seq = [int(x) for x in picked_slots if int(x) >= 0][:k]
    if len(seq) != k:
        raise ValueError(f"need {k} valid picks, got {seq!r}")
    if len(set(seq)) != len(seq):
        raise ValueError(f"picks must be unique, got {seq!r}")
    rest = sorted(set(range(BOARD_SIZE)) - set(seq))
    return tuple(seq + rest)


def validate_struct_action(a: StructAction) -> None:
    t = a.type
    if t == StructActionType.ROLL or t == StructActionType.LEVEL_UP:
        if a.args != ():
            raise ValueError(f"{t.name} expects args (), got {a.args}")
    elif t == StructActionType.BUY:
        if len(a.args) != 1:
            raise ValueError(f"BUY expects args (shop_slot,), got {a.args}")
        s = a.args[0]
        if s < 0 or s >= SHOP_SIZE:
            raise ValueError(f"BUY shop_slot out of range [0,{SHOP_SIZE}): {s}")
    elif t == StructActionType.SELL:
        if len(a.args) != 1:
            raise ValueError(f"SELL expects args (board_slot,), got {a.args}")
        p = a.args[0]
        if p < 0 or p >= BOARD_SIZE:
            raise ValueError(f"SELL board_slot out of range [0,{BOARD_SIZE}): {p}")
    elif t == StructActionType.PLACE:
        if len(a.args) != 1:
            raise ValueError(f"PLACE expects args (hand_slot,), got {a.args}")
        h = a.args[0]
        if h < 0 or h >= HAND_SIZE:
            raise ValueError(f"PLACE hand_slot out of range [0,{HAND_SIZE}): {h}")
    elif t == StructActionType.COMPLETE_TURN:
        if a.args != ():
            raise ValueError(f"COMPLETE_TURN expects args (); pass board_perm to env.step_structured")
    else:
        raise ValueError(f"unknown StructActionType {t}")


def structured_legal_set(actions: Tuple[StructAction, ...]) -> FrozenSet[StructAction]:
    return frozenset(actions)


def structured_action_to_replay_env_int(a: StructAction) -> int:
    """Approximate mapping for JSONL replay ``a`` field (legacy decoders)."""
    from .action_map import (
        A_BUY_BASE,
        A_FINISH,
        A_LEVEL_UP,
        A_PLACE_BASE,
        A_ROLL,
        A_SELL_BASE,
    )

    if a.type == StructActionType.ROLL:
        return int(A_ROLL)
    if a.type == StructActionType.LEVEL_UP:
        return int(A_LEVEL_UP)
    if a.type == StructActionType.BUY:
        return int(A_BUY_BASE + a.args[0])
    if a.type == StructActionType.SELL:
        return int(A_SELL_BASE + a.args[0])
    if a.type == StructActionType.PLACE:
        return int(A_PLACE_BASE + a.args[0])
    if a.type == StructActionType.COMPLETE_TURN:
        return int(A_FINISH)
    raise ValueError(a)


__all__ = [
    "BOARD_SIZE",
    "StructAction",
    "StructActionType",
    "structured_action_to_replay_env_int",
    "structured_legal_set",
    "slot_pick_sequence_to_perm",
    "validate_board_perm",
    "validate_struct_action",
]
