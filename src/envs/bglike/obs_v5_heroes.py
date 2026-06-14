"""v5-heroes observation: ``obs_v5`` + a hero-power block at the tail.

Layout (strict superset of :mod:`src.envs.bglike.obs_v5`):

    [ build_observation_v5(...)         — OBS_DIM_V5 floats ]
    [ self-hero features                — HERO_SELF_DIM floats ]
    [ opponent-hero one-hots            — MAX_OPPS * NUM_HERO_OBS_IDS floats ]

Self-hero features (HERO_SELF_DIM = 35):
    identity one-hot (NUM_HERO_OBS_IDS) | effective roll/buy/level-up cost (3) |
    rotating-tribe one-hot (RACE_ONEHOT_DIM, Rat King) | Kael'thas buy progress |
    Chenvaala elemental progress | Chenvaala accumulated discount |
    Nozdormu free-roll flag | A.F. Kay rounds-left | Deathwing aura |
    Al'Akir start-of-combat flag.

The **effective costs** intentionally read ``economy.effective_*`` (not the raw
fields the base obs uses) so Millhouse / Nozdormu / Chenvaala are observable —
the base obs core misreports cost under hero modifiers. Opponent heroes are
emitted in the SAME HP-sorted order as the opponent panel (``sorted_opponent_rows``)
so the v11_heroes net can fuse each one-hot into the matching opponent token.

Used only with ``obs_kind="bglike_v5_heroes"`` + the ``bglike_structured_v11_heroes``
net. With ``with_heroes=False`` every seat's hero is None, so the block degrades
to the "no hero" bucket (identity index 0, base costs) — still fixed width.
"""

from __future__ import annotations

import numpy as np

from src.bg_catalog.patch_context import PatchContext
from src.bg_core.hero import (
    EveryNthBuyBuff,
    UpgradeDiscountPerElementals,
    ZeroGoldForRounds,
)
from src.bg_lobby.player import PlayerState
from src.bg_recruitment import economy
from src.envs.minibg.obs import RACE_ONEHOT_DIM, _RACE_ORDER

from .actions import LEVEL_UP_COSTS
from .obs import MAX_OPPS, sorted_opponent_rows
from .obs_v5 import OBS_DIM_V5, build_observation_v5
from .state import BGLikeState

# 15 patch-19.6 heroes + index 0 = none/unknown. Fixed so the obs width is a
# constant (heroes beyond 15 in a future patch collapse into the last bucket).
NUM_HERO_OBS_IDS = 16

_LEVEL_UP_COST_MAX = max(LEVEL_UP_COSTS.values())

# Self-hero sub-block sizes / offsets.
_IDENT_DIM = NUM_HERO_OBS_IDS          # 16
_COST_DIM = 3                          # eff roll / buy / level-up
_ROT_DIM = RACE_ONEHOT_DIM             # 9  (rotating tribe; index 0 == None)
_DYN_DIM = 5                           # kael, chenvaala-elem, chenvaala-disc, nozdormu, afk
_COMBAT_DIM = 2                        # deathwing aura, al'akir flag
HERO_SELF_DIM = _IDENT_DIM + _COST_DIM + _ROT_DIM + _DYN_DIM + _COMBAT_DIM  # 35

HERO_OPP_DIM = MAX_OPPS * NUM_HERO_OBS_IDS  # 7 * 16 = 112
HERO_BLOCK_DIM = HERO_SELF_DIM + HERO_OPP_DIM  # 147
OBS_DIM_V5_HEROES = OBS_DIM_V5 + HERO_BLOCK_DIM

HERO_BLOCK_OFFSET = OBS_DIM_V5
HERO_SELF_OFFSET = HERO_BLOCK_OFFSET
HERO_OPP_OFFSET = HERO_SELF_OFFSET + HERO_SELF_DIM


def hero_obs_index(patch: PatchContext, hero) -> int:
    """Dense hero index for the one-hot: 0 = none/unknown, 1..N = sorted pool."""
    if hero is None:
        return 0
    pool = sorted(patch.hero_pool_ids)
    try:
        return min(pool.index(hero.hero_id) + 1, NUM_HERO_OBS_IDS - 1)
    except ValueError:
        return 0


def _one_hot(idx: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    if 0 <= idx < n:
        v[idx] = 1.0
    return v


def _race_index(race) -> int:
    try:
        return _RACE_ORDER.index(race)
    except ValueError:
        return 0


def _self_hero_features(state: BGLikeState, me: PlayerState, patch: PatchContext) -> np.ndarray:
    out = np.zeros(HERO_SELF_DIM, dtype=np.float32)
    h = me.hero

    i = 0
    out[i : i + _IDENT_DIM] = _one_hot(hero_obs_index(patch, h), NUM_HERO_OBS_IDS)
    i += _IDENT_DIM

    # Effective costs — capture Millhouse / Nozdormu / Chenvaala directly.
    out[i] = economy.effective_roll_cost(me) / 3.0
    out[i + 1] = economy.effective_buy_cost(me) / 5.0
    out[i + 2] = economy.effective_level_up_cost(me) / float(_LEVEL_UP_COST_MAX)
    i += _COST_DIM

    # Rotating tribe (Rat King).
    if me.hero_rotating_tribe is not None:
        out[i : i + _ROT_DIM] = _one_hot(_race_index(me.hero_rotating_tribe), _ROT_DIM)
    i += _ROT_DIM

    # Dynamics. Offsets within the 5-wide block:
    #   0 kael progress | 1 chenvaala elem progress | 2 chenvaala discount |
    #   3 nozdormu free-roll | 4 a.f.kay rounds-left
    if h is not None:
        for p in h.passives:
            if isinstance(p, EveryNthBuyBuff) and p.n > 0:
                out[i + 0] = float(me.hero_buy_count % p.n) / float(p.n)
            elif isinstance(p, UpgradeDiscountPerElementals) and p.per > 0:
                out[i + 1] = float(me.hero_elementals_progress) / float(p.per)
            elif isinstance(p, ZeroGoldForRounds):
                left = sum(1 for r in p.rounds if r >= state.round_number)
                out[i + 4] = float(left) / 2.0
        out[i + 2] = float(me.hero_upgrade_discount) / 10.0
        out[i + 3] = 1.0 if me.hero_free_roll_pending else 0.0
    i += _DYN_DIM

    # Combat-only effects (not visible in shop state).
    if h is not None:
        out[i] = float(h.combat_attack_aura()) / 5.0
        out[i + 1] = 1.0 if h.start_combat_leftmost_keywords() else 0.0
    return out


def _opp_hero_features(state: BGLikeState, seat: int, patch: PatchContext) -> np.ndarray:
    out = np.zeros((MAX_OPPS, NUM_HERO_OBS_IDS), dtype=np.float32)
    rows = sorted_opponent_rows(state, seat)
    for j, row in enumerate(rows[:MAX_OPPS]):
        opp_seat = row[0]
        out[j] = _one_hot(
            hero_obs_index(patch, state.players[opp_seat].hero), NUM_HERO_OBS_IDS
        )
    return out.reshape(-1)


def build_observation_v5_heroes(
    state: BGLikeState,
    seat: int,
    last_battle_signed: float,
    *,
    is_my_turn: bool,
    patch: PatchContext,
    rl_pending=None,
) -> np.ndarray:
    """Return v5-heroes obs (obs_v5 + self/opponent hero block at tail)."""
    base = build_observation_v5(
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
            _self_hero_features(state, me, patch),
            _opp_hero_features(state, seat, patch),
        ]
    )


__all__ = [
    "NUM_HERO_OBS_IDS",
    "HERO_SELF_DIM",
    "HERO_OPP_DIM",
    "HERO_BLOCK_DIM",
    "OBS_DIM_V5_HEROES",
    "HERO_BLOCK_OFFSET",
    "HERO_SELF_OFFSET",
    "HERO_OPP_OFFSET",
    "hero_obs_index",
    "build_observation_v5_heroes",
]
