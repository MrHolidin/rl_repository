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


_HERO_NETWORK_TYPES = frozenset({"bglike_structured_v11_heroes"})


def validate_heroes_consistency(game_id: str, network_type: str, game_params: dict) -> None:
    """Hard-fail on a heroes/network mismatch.

    The hero obs (``bglike_v5_heroes``) and a hero-aware net must be used
    together: a non-hero net can't observe the hero block, and the hero net's
    obs is meaningless (and wrongly shaped vs the assigned heroes) without
    ``with_heroes``. Either mistake silently wastes a run, so reject it early.
    """
    gid = (game_id or "").strip().lower()
    if gid != "bglike":
        return
    nt = (network_type or "").strip().lower()
    is_hero_net = nt in _HERO_NETWORK_TYPES
    with_heroes = bool(game_params.get("with_heroes", False))
    if with_heroes and not is_hero_net:
        raise ValueError(
            f"game.params.with_heroes=true but network_type={nt!r} cannot observe "
            f"heroes; use network_type=bglike_structured_v11_heroes (or set with_heroes=false)."
        )
    if is_hero_net and not with_heroes:
        raise ValueError(
            f"network_type={nt!r} observes the hero block but game.params.with_heroes "
            f"is false; set with_heroes=true (or use a non-hero net)."
        )


__all__ = ["reject_flat_bg_network", "validate_heroes_consistency"]
