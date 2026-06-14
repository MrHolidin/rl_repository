"""Default observation vector sizing for training entrypoints."""

from __future__ import annotations

from typing import Any, Dict, Optional

_BG_GAME_IDS = frozenset({"minibg", "bglike"})


def apply_bg_observation_defaults(
    game_id: str,
    agent_params: Dict[str, Any],
    *,
    obs_kind: Optional[str] = None,
) -> None:
    """Set flat-vector ``observation_shape`` / ``observation_type`` from obs layout.

    For bglike, ``obs_kind`` picks between the default obs and ``"bglike_v5"``
    (the per-ability-token superset consumed by ``bglike_structured_v5``).
    """
    gid = (game_id or "").strip().lower()
    if gid not in _BG_GAME_IDS:
        return
    if gid == "bglike":
        kind = (obs_kind or "bglike").strip().lower()
        if kind == "bglike_v5_heroes":
            from src.envs.bglike.obs_v5_heroes import OBS_DIM_V5_HEROES as OBS_DIM
        elif kind == "bglike_v5":
            from src.envs.bglike.obs_v5 import OBS_DIM_V5 as OBS_DIM
        else:
            from src.envs.bglike.obs import OBS_DIM
    else:
        from src.envs.minibg.obs import OBS_DIM

    agent_params.setdefault("observation_shape", (OBS_DIM,))
    agent_params.setdefault("observation_type", "vector")


__all__ = ["apply_bg_observation_defaults"]
