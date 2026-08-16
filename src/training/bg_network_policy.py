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


_HERO_NETWORK_TYPES = frozenset(
    {"bglike_structured_v11_heroes", "bglike_structured_v12", "bglike_structured_v13"}
)


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
            f"heroes; use one of {sorted(_HERO_NETWORK_TYPES)} (or set with_heroes=false)."
        )
    if is_hero_net and not with_heroes:
        raise ValueError(
            f"network_type={nt!r} observes the hero block but game.params.with_heroes "
            f"is false; set with_heroes=true (or use a non-hero net)."
        )


__all__ = ["reject_flat_bg_network", "validate_heroes_consistency"]


# --------------------------------------------------------------------------- #
# Which observation each network reads
# --------------------------------------------------------------------------- #

# Single source of truth. The training entrypoints auto-pin game.params.obs_kind
# from this (a config that disagrees is rejected rather than silently reshaped),
# and evaluation uses it to work out what each checkpoint expects -- which is
# what lets one lobby host checkpoints of different versions.
#
# Networks absent from this table read the base ``bglike`` obs.
NETWORK_OBS_KIND: dict[str, str] = {
    "bglike_structured_v5": "bglike_v5",
    "bglike_structured_v6": "bglike_v5",
    "bglike_structured_v7": "bglike_v5",
    "bglike_structured_v8": "bglike_v5",
    "bglike_structured_v9": "bglike_v5",
    "bglike_structured_v10": "bglike_v5",
    "bglike_structured_v11": "bglike_v5",
    "bglike_structured_v11_heroes": "bglike_v5_heroes",
    "bglike_structured_v12": "bglike_v6_heroes",
    "bglike_structured_v13": "bglike_v7_pref",
}

# Networks that additionally require heroes to be dealt.
NETWORK_REQUIRES_HEROES = frozenset(_HERO_NETWORK_TYPES)


def obs_kind_for_network(network_type: str, *, default: str = "bglike") -> str:
    """The obs layout ``network_type`` reads. Unknown networks get ``default``."""
    return NETWORK_OBS_KIND.get((network_type or "").strip().lower(), default)


def obs_kind_for_checkpoint(path) -> str:
    """Read a checkpoint's ``ppo_network_type`` and map it to its obs layout.

    Lets an evaluation script mix checkpoints without being told which
    observation each one wants.
    """
    import torch

    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    return obs_kind_for_network(str(ck.get("ppo_network_type", "")))
