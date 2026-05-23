"""Battlegrounds RL network policy (flat deprecated, structured required)."""

from __future__ import annotations

_FLAT_NETWORK_TYPES = frozenset({"minibg_mlp", "flat_mlp", "mlp", "dueling_dqn"})
_BG_GAME_IDS = frozenset({"minibg", "bglike"})


def reject_flat_bg_network(
    game_id: str,
    network_type: str,
    *,
    agent_id: str | None = None,
) -> None:
    """Raise if a flat vector policy is requested for a Battlegrounds ruleset."""
    gid = (game_id or "").strip().lower()
    nt = (network_type or "").strip().lower()
    if gid not in _BG_GAME_IDS:
        return
    if nt not in _FLAT_NETWORK_TYPES:
        return
    who = f"agent.id={agent_id} " if agent_id else ""
    raise ValueError(
        f"{who}Flat PPO/DQN is deprecated for Battlegrounds ({gid}). "
        "Use network_type: minibg_structured or bglike_structured."
    )


__all__ = ["reject_flat_bg_network"]
