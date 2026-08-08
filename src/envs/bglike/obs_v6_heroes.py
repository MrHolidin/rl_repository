"""v6-heroes observation: base bglike obs + hero block, **no ability tail**.

This is ``obs_v5_heroes`` with the 1560-float per-ability block removed::

    v5_heroes = [ base 976 ] [ abilities 1560 ] [ hero 147 ]   = 2683
    v6_heroes = [ base 976 ]                    [ hero 147 ]   = 1123   (−58%)

The ability block re-encoded, on every observation, facts that are fixed
properties of a *card template*: which effect classes it carries, their
triggers, their numeric parameters. Those now live in a frozen table the net
gathers by the ``card_idx`` channel the slot vector already carries (see
:mod:`src.envs.bglike.card_static`) — rules-text embedding for the mechanic,
DSL fields for the magnitudes.

What deliberately stays in the observation, because it is *runtime* state and
a template-keyed table cannot know it:

* keywords and divine shield — granted by buffs, lost mid-combat, so the
  printed keywords of the card are the wrong answer;
* current attack / health, base and bonus split;
* tavern tier — card text does not mention it at all (a linear probe on the
  text embedding scores *below* the majority baseline for tier, 0.150 vs
  0.213), while the swap probe rates tier the single strongest channel;
* golden, frozen, triple-reward, same-card counts, position.

Known blind spot, accepted for this version: a magnetised mech carries
abilities merged in from another card, so its ``card_id`` no longer describes
it. See :func:`src.envs.bglike.card_static.magnetic_divergence_note`.

Used with ``obs_kind="bglike_v6_heroes"`` + ``bglike_structured_v12``.
"""

from __future__ import annotations

import numpy as np

from src.bg_catalog.patch_context import PatchContext

from .obs import OBS_DIM, build_observation
from .obs_v5_heroes import (
    HERO_BLOCK_DIM,
    HERO_OPP_DIM,
    HERO_SELF_DIM,
    opp_hero_features,
    self_hero_features,
)
from .state import BGLikeState

OBS_DIM_V6_HEROES = OBS_DIM + HERO_BLOCK_DIM  # 976 + 147 = 1123

HERO_BLOCK_OFFSET_V6 = OBS_DIM
HERO_SELF_OFFSET_V6 = HERO_BLOCK_OFFSET_V6
HERO_OPP_OFFSET_V6 = HERO_SELF_OFFSET_V6 + HERO_SELF_DIM


def build_observation_v6_heroes(
    state: BGLikeState,
    seat: int,
    last_battle_signed: float,
    *,
    is_my_turn: bool,
    patch: PatchContext,
    rl_pending=None,
) -> np.ndarray:
    """Return the v6-heroes obs (base bglike obs + self/opponent hero block)."""
    base = build_observation(
        state,
        seat,
        last_battle_signed,
        is_my_turn=is_my_turn,
        patch=patch,
        rl_pending=rl_pending,
    )
    me = state.players[seat]
    return np.concatenate(
        [
            base,
            self_hero_features(state, me, patch),
            opp_hero_features(state, seat, patch),
        ]
    )


__all__ = [
    "OBS_DIM_V6_HEROES",
    "HERO_BLOCK_DIM",
    "HERO_SELF_DIM",
    "HERO_OPP_DIM",
    "HERO_BLOCK_OFFSET_V6",
    "HERO_SELF_OFFSET_V6",
    "HERO_OPP_OFFSET_V6",
    "build_observation_v6_heroes",
]
