"""Shop-phase invariants: detect impossible empty legal (softlock) with a visible report."""

from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from src.envs.minibg.state import MiniBGState, PendingChoiceKind, PlayerPhase, PlayerState

_EMPTY_LEGAL_LOG = "empty_legal_assertion.log"
_ILLEGAL_ACTION_LOG = "illegal_action_assertion.log"


def _hand_slots(player: PlayerState) -> list[str | None]:
    return [m.card_id if m is not None else None for m in player.hand]


def format_empty_legal_report(state: MiniBGState, *, where: str) -> str:
    """Human-readable dump when shop ``legal_actions`` is empty but the game is not over."""
    player = state.players[state.current_player_index]
    pc = player.pending_choice
    lines = [
        "=" * 72,
        f"EMPTY_SHOP_LEGAL @ {where}",
        f"time_utc={datetime.now(timezone.utc).isoformat()}",
        f"pid={os.getpid()}",
        f"round={state.round_number} current_player={state.current_player_index}",
        f"phase={player.phase.name} done={state.done}",
        f"gold={player.gold} tier={player.tavern_tier} shop_actions_used={player.shop_actions_used}",
        f"board_len={len(player.board)} hand={_hand_slots(player)}",
        f"hand_filled={sum(1 for m in player.hand if m is not None)}/5",
    ]
    if pc is not None:
        lines.append(
            f"pending_choice kind={pc.kind.name} extra_modals_after={pc.extra_modals_after}"
        )
        lines.append(f"pending_options={pc.options!r}")
    else:
        lines.append("pending_choice=None")
    lines.append(
        f"placed_minion_pending_after={getattr(player.placed_minion_pending_after, 'card_id', None)}"
    )
    lines.append(
        f"triple_reward_discover_pending={player.triple_reward_discover_pending} "
        f"spell_tier={player.triple_reward_spell_tier}"
    )
    lines.append(f"stack:\n{''.join(traceback.format_stack(limit=12))}")
    lines.append("=" * 72)
    return "\n".join(lines)


def log_diagnostic_report(msg: str, log_name: str) -> None:
    print(msg, file=sys.stderr, flush=True)
    run_dir = os.environ.get("RL_RUN_DIR", "").strip()
    if not run_dir:
        return
    path = Path(run_dir) / log_name
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(msg)
            f.write("\n")
    except OSError as e:
        print(f"could not write {path}: {e}", file=sys.stderr, flush=True)


def log_empty_legal_report(msg: str) -> None:
    log_diagnostic_report(msg, _EMPTY_LEGAL_LOG)


def assert_shop_has_legal_actions(
    state: MiniBGState,
    legal: Sequence[int],
    *,
    where: str,
) -> None:
    """Raise if the shop has no legal actions while the episode is still live in shop."""
    if state.done:
        return
    player = state.players[state.current_player_index]
    if player.phase != PlayerPhase.SHOP:
        return
    if legal:
        return
    msg = format_empty_legal_report(state, where=where)
    log_empty_legal_report(msg)
    raise RuntimeError(
        f"EMPTY_SHOP_LEGAL at {where} (see stderr and "
        f"{os.environ.get('RL_RUN_DIR', '.')}/{_EMPTY_LEGAL_LOG})"
    )


def format_illegal_action_report(
    state: MiniBGState,
    *,
    where: str,
    action: Optional[int] = None,
    mask: Optional[np.ndarray] = None,
    structured_action: Optional[str] = None,
    detail: str = "",
    rl_pending: bool = False,
) -> str:
    player = state.players[state.current_player_index]
    pc = player.pending_choice
    mask_allows = None
    if action is not None and mask is not None:
        mask_allows = bool(0 <= action < len(mask) and mask[action])
    legal_count = int(mask.sum()) if mask is not None else None
    lines = [
        "=" * 72,
        f"ILLEGAL_ACTION @ {where}",
        f"time_utc={datetime.now(timezone.utc).isoformat()}",
        f"pid={os.getpid()}",
        f"action={action} structured_action={structured_action!r}",
        f"mask_allows={mask_allows} legal_mask_count={legal_count}",
        f"rl_pending={rl_pending} detail={detail!r}",
        f"round={state.round_number} current_player={state.current_player_index}",
        f"phase={player.phase.name} done={state.done}",
        f"gold={player.gold} tier={player.tavern_tier}",
        f"board_len={len(player.board)} hand={_hand_slots(player)}",
    ]
    if pc is not None:
        lines.append(f"pending_choice kind={pc.kind.name}")
    lines.append(f"stack:\n{''.join(traceback.format_stack(limit=12))}")
    lines.append("=" * 72)
    return "\n".join(lines)


def format_illegal_mask_report(
    state: MiniBGState,
    action: int,
    mask: np.ndarray,
    *,
    where: str,
    rl_pending: bool = False,
) -> str:
    return format_illegal_action_report(
        state,
        where=where,
        action=action,
        mask=mask,
        rl_pending=rl_pending,
        detail="not in legal_actions_mask",
    )


def raise_illegal_env_action(
    state: MiniBGState,
    *,
    where: str,
    action: Optional[int] = None,
    mask: Optional[np.ndarray] = None,
    structured_action: Optional[str] = None,
    detail: str = "",
    rl_pending: bool = False,
) -> None:
    """Log and raise on any illegal env / structured action (never soft-invalid)."""
    msg = format_illegal_action_report(
        state,
        where=where,
        action=action,
        mask=mask,
        structured_action=structured_action,
        detail=detail,
        rl_pending=rl_pending,
    )
    log_diagnostic_report(msg, _ILLEGAL_ACTION_LOG)
    raise RuntimeError(
        f"ILLEGAL_ACTION at {where} "
        f"(see stderr and {os.environ.get('RL_RUN_DIR', '.')}/{_ILLEGAL_ACTION_LOG})"
    )


def assert_action_in_legal_mask(
    state: MiniBGState,
    action: int,
    mask: np.ndarray,
    *,
    where: str,
    rl_pending: Optional[bool] = None,
) -> None:
    """Raise if ``action`` is not allowed by ``legal_actions_mask``."""
    if state.done:
        return
    if 0 <= action < len(mask) and bool(mask[action]):
        return
    raise_illegal_env_action(
        state,
        where=where,
        action=action,
        mask=mask,
        detail="not in legal_actions_mask",
        rl_pending=bool(rl_pending),
    )


__all__ = [
    "assert_action_in_legal_mask",
    "assert_shop_has_legal_actions",
    "format_empty_legal_report",
    "format_illegal_action_report",
    "format_illegal_mask_report",
    "log_diagnostic_report",
    "log_empty_legal_report",
    "raise_illegal_env_action",
]
