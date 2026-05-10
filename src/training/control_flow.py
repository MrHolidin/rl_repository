"""Pluggable control flow: who acts next after strict alternation vs multi-step phases."""

from __future__ import annotations

from typing import Literal, Optional, Protocol, runtime_checkable

from src.envs.base import TurnBasedEnv

ActingRole = Literal["agent", "opponent"]


@runtime_checkable
class ControlDriver(Protocol):
    """Decides whose turn it is and how agent TD transitions are credited."""

    @property
    def uses_paired_credit(self) -> bool:
        """If True: one agent step pairs with exactly one opponent step (Connect4-style)."""

    def active_role(self, env: TurnBasedEnv, agent_token: int) -> ActingRole:
        """Who must act next (call only when not env.done)."""


class AlternatingDriver:
    """Strict two-player alternation via ``current_player_token`` (fallback: P0 = token +1)."""

    @property
    def uses_paired_credit(self) -> bool:
        return True

    def active_role(self, env: TurnBasedEnv, agent_token: int) -> ActingRole:
        tok = getattr(env, "current_player_token", None)
        if tok is None:
            return "agent" if agent_token == 1 else "opponent"
        return "agent" if int(tok) == agent_token else "opponent"


class MiniBGDriver:
    """MiniBG shop: many consecutive env.steps per player before the other acts."""

    @property
    def uses_paired_credit(self) -> bool:
        return False

    def active_role(self, env: TurnBasedEnv, agent_token: int) -> ActingRole:
        tok = getattr(env, "current_player_token", None)
        if tok is None:
            raise TypeError("MiniBGDriver requires env.current_player_token")
        return "agent" if int(tok) == agent_token else "opponent"


_DRIVER_REGISTRY: dict[str, type] = {
    "alternating": AlternatingDriver,
    "minibg": MiniBGDriver,
}


def register_control_driver(driver_id: str, cls: type) -> None:
    if driver_id in _DRIVER_REGISTRY:
        raise ValueError(f"control driver {driver_id!r} already registered")
    _DRIVER_REGISTRY[driver_id] = cls


def make_control_driver(game_id: str, driver_id: Optional[str] = None) -> ControlDriver:
    gid = (game_id or "").strip().lower()
    if driver_id is None or str(driver_id).strip() == "":
        resolved = "minibg" if gid == "minibg" else "alternating"
    else:
        resolved = str(driver_id).strip().lower()
    cls = _DRIVER_REGISTRY.get(resolved)
    if cls is None:
        known = ", ".join(sorted(_DRIVER_REGISTRY))
        raise KeyError(f"Unknown control_driver {resolved!r}; known: {known}")
    return cls()


__all__ = [
    "ActingRole",
    "ControlDriver",
    "AlternatingDriver",
    "MiniBGDriver",
    "register_control_driver",
    "make_control_driver",
]
