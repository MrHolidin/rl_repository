"""Default observation vector sizing for training entrypoints."""

from __future__ import annotations

from typing import Any, Dict

_BG_GAME_IDS = frozenset({"minibg", "bglike"})


def apply_bg_observation_defaults(game_id: str, agent_params: Dict[str, Any]) -> None:
    """Set flat-vector ``observation_shape`` / ``observation_type`` from obs layout."""
    gid = (game_id or "").strip().lower()
    if gid not in _BG_GAME_IDS:
        return
    if gid == "bglike":
        from src.envs.bglike.obs import OBS_DIM
    else:
        from src.envs.minibg.obs import OBS_DIM

    agent_params.setdefault("observation_shape", (OBS_DIM,))
    agent_params.setdefault("observation_type", "vector")


__all__ = ["apply_bg_observation_defaults"]
