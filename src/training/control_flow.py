"""Tiny helper for ``current_player_token``-based turn dispatch.

Historically this module distinguished paired (Connect4) vs multi-step
(MiniBG) turn semantics for the trainer; the trainer now consumes a
`SingleAgentEnv` and that distinction is gone. The remaining surface is
the `active_role` predicate, kept as a stable utility for off-trainer
scripts that drive a raw `TurnBasedEnv` themselves (e.g. eval/seat-swap
harnesses).
"""

from __future__ import annotations

from typing import Literal

from src.envs.base import TurnBasedEnv

ActingRole = Literal["agent", "opponent"]


def active_role(env: TurnBasedEnv, agent_token: int) -> ActingRole:
    """Return ``"agent"`` if it is currently the agent's turn, else ``"opponent"``."""
    tok = getattr(env, "current_player_token", None)
    if tok is None:
        return "agent" if agent_token == 1 else "opponent"
    return "agent" if int(tok) == agent_token else "opponent"


__all__ = ["ActingRole", "active_role"]
