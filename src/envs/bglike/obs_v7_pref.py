"""v7 = v6_heroes + the seat's own tribe-preference vector.

    obs = [ base(976) | hero block(147) | tribe pref(7) ] = 1130

Only the acting seat's own vector is exposed: the preferences are private, so
a seat cannot read what its opponents are being paid to collect. A seat whose
vector was never drawn (``with_tribe_pref`` off) reports zeros, which is also
what "no preference" means to the shaping term.
"""

from __future__ import annotations

import numpy as np

from src.bg_catalog.patch_context import PatchContext

from .obs_v6_heroes import OBS_DIM_V6_HEROES, build_observation_v6_heroes
from .state import BGLikeState
from .tribe_pref import NUM_TRIBES

TRIBE_PREF_DIM = NUM_TRIBES
OBS_DIM_V7_PREF = OBS_DIM_V6_HEROES + TRIBE_PREF_DIM  # 1123 + 7 = 1130
TRIBE_PREF_OFFSET_V7 = OBS_DIM_V6_HEROES


def tribe_pref_features(state: BGLikeState, seat: int) -> np.ndarray:
    pref = getattr(state.players[seat], "tribe_pref", ()) or ()
    out = np.zeros(TRIBE_PREF_DIM, dtype=np.float32)
    n = min(len(pref), TRIBE_PREF_DIM)
    if n:
        out[:n] = np.asarray(pref[:n], dtype=np.float32)
    return out


def build_observation_v7_pref(
    state: BGLikeState,
    seat: int,
    last_battle_signed: float,
    *,
    is_my_turn: bool,
    patch: PatchContext,
    rl_pending=None,
) -> np.ndarray:
    base = build_observation_v6_heroes(
        state,
        seat,
        last_battle_signed,
        is_my_turn=is_my_turn,
        patch=patch,
        rl_pending=rl_pending,
    )
    return np.concatenate([base, tribe_pref_features(state, seat)])


__all__ = [
    "OBS_DIM_V7_PREF",
    "TRIBE_PREF_DIM",
    "TRIBE_PREF_OFFSET_V7",
    "build_observation_v7_pref",
    "tribe_pref_features",
]
