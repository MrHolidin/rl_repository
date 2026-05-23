"""Unified agent→env step dispatch for lobby seat controllers and single-agent envs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from src.envs.bglike.action_map import struct_action_to_log_int
from src.envs.minibg.structured_actions import StructAction, StructActionType

if TYPE_CHECKING:
    from src.agents.base_agent import BaseAgent
    from src.envs.bglike.lobby_env import BGLobbyEnv, LobbyStepInfo


class _LobbyStateView:
    """Minimal handle for ``act_structured`` during per-seat lobby control."""

    def __init__(self, lobby: BGLobbyEnv) -> None:
        self._lobby = lobby

    @property
    def state(self):
        return self._lobby.state


@dataclass(frozen=True)
class LobbySeatStepResult:
    info: LobbyStepInfo
    action: int
    controller: str = ""
    control_path: str = ""
    struct_action: str = ""


def supports_structured_lobby_control(agent: Any, lobby: Any) -> bool:
    return (
        callable(getattr(agent, "act_structured", None))
        and callable(getattr(lobby, "legal_structured_actions_for_seat", None))
        and callable(getattr(lobby, "step_structured_for_seat", None))
    )


def control_path_for_agent(agent: Any, lobby: Any) -> str:
    return "structured" if supports_structured_lobby_control(agent, lobby) else "flat"


def describe_seat_controller(
    agent: Any,
    *,
    seat: int,
    lobby: Any,
    controller_env: Any = None,
) -> str:
    """Compact label for drain debug (class, heuristic name, league slot, object id)."""
    parts: list[str] = [type(agent).__name__]
    bot = getattr(agent, "_bot", None)
    bot_name = getattr(bot, "name", None)
    if bot_name:
        parts.append(str(bot_name))
    slot_map = getattr(controller_env, "_opponent_slot_by_seat", None) or {}
    slot_id = slot_map.get(seat)
    if slot_id is not None:
        from src.training.selfplay.league_state import SLOT_CURRENT, SLOT_SCRIPTED

        if slot_id == SLOT_SCRIPTED:
            parts.append("slot=SCRIPTED")
        elif slot_id == SLOT_CURRENT:
            parts.append("slot=CURRENT")
        else:
            parts.append(f"slot=FROZEN#{slot_id}")
    parts.append(f"id={id(agent)}")
    return "/".join(parts)


def lobby_seat_step(
    lobby: BGLobbyEnv,
    seat: int,
    agent: BaseAgent,
    *,
    deterministic: bool = False,
    controller_env: Any = None,
) -> LobbySeatStepResult:
    """Apply one controller decision for ``seat`` (structured or flat)."""
    if lobby.state.current_player_index != seat:
        raise RuntimeError(
            f"lobby_seat_step: seat {seat} is not current "
            f"(current={lobby.state.current_player_index})"
        )

    controller = describe_seat_controller(
        agent,
        seat=seat,
        lobby=lobby,
        controller_env=controller_env,
    )
    control_path = control_path_for_agent(agent, lobby)

    obs = lobby.obs_for_seat(seat)
    mask = lobby.legal_mask_for_seat(seat)
    if not bool(mask.any()):
        raise RuntimeError(f"no legal actions for seat {seat}")

    lobby._heuristic_control_seat = seat
    try:
        if supports_structured_lobby_control(agent, lobby):
            legal = lobby.legal_structured_actions_for_seat(seat)
            if not legal:
                raise RuntimeError(f"structured legal set empty for seat {seat}")
            ctx = _LobbyStateView(lobby)
            was_training = bool(getattr(agent, "training", False))
            if hasattr(agent, "training"):
                agent.training = False
            try:
                struct_act, board_perm, _idx = agent.act_structured(
                    obs,
                    legal,
                    ctx,
                    deterministic=deterministic,
                )
            finally:
                if hasattr(agent, "training"):
                    agent.training = was_training
            info = lobby.step_structured_for_seat(
                seat,
                struct_act,
                board_perm=board_perm,
            )
            return LobbySeatStepResult(
                info=info,
                action=struct_action_to_log_int(struct_act),
                controller=controller,
                control_path=control_path,
                struct_action=str(struct_act.type.name),
            )

        if controller_env is not None and hasattr(agent, "set_env"):
            agent.set_env(controller_env)
        action = int(agent.act(obs, legal_mask=mask, deterministic=deterministic))
        info = lobby._apply_action(seat, action, auto=True)
        return LobbySeatStepResult(
            info=info,
            action=action,
            controller=controller,
            control_path=control_path,
        )
    finally:
        lobby._heuristic_control_seat = None


__all__ = [
    "LobbySeatStepResult",
    "control_path_for_agent",
    "describe_seat_controller",
    "lobby_seat_step",
    "supports_structured_lobby_control",
]
