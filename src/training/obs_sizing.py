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

    For bglike, ``obs_kind`` picks the layout: the default obs, ``"bglike_v5"``
    (per-ability-token superset), ``"bglike_v5_heroes"`` (+ hero block), or
    ``"bglike_v6_heroes"`` (base + hero block, ability tail dropped — v12 reads
    card facts from a frozen table instead of the observation), or
    ``"bglike_v7_pref"`` (v6 + the seat's own tribe-preference vector — v13).
    """
    gid = (game_id or "").strip().lower()
    if gid not in _BG_GAME_IDS:
        return
    if gid == "bglike":
        kind = (obs_kind or "bglike").strip().lower()
        if kind == "bglike_v7_pref":
            from src.envs.bglike.obs_v7_pref import OBS_DIM_V7_PREF as OBS_DIM
        elif kind == "bglike_v6_heroes":
            from src.envs.bglike.obs_v6_heroes import OBS_DIM_V6_HEROES as OBS_DIM
        elif kind == "bglike_v5_heroes":
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
