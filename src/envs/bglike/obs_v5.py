"""BGLike v5 observation: v3 obs + per-ability token block appended at tail.

Layout (strict superset of :mod:`src.envs.bglike.obs`):
    [ <existing build_observation(...) bytes — OBS_DIM floats> ]
    [ own_abilities    : BOARD_SIZE     * K_ABIL * ABIL_FEAT_DIM floats ]
    [ shop_abilities   : MAX_SHOP_SLOTS * K_ABIL * ABIL_FEAT_DIM floats ]
    [ hand_abilities   : HAND_SIZE      * K_ABIL * ABIL_FEAT_DIM floats ]
    [ pending_abilities: PENDING_LEN    * K_ABIL * ABIL_FEAT_DIM floats ]

Per ability token = ``ABIL_FEAT_DIM`` floats packed as
``(effect_id+1, trigger_id+1, condition_kind_id+1, condition_arg_race_id+1,
filter_race_id+1, combat_only, atk, hp, amount, repeats, count)``.
The ``+1`` shift makes ``0`` the **padding / "no value"** id so the model can
use a single ``Embedding(N+1, d, padding_idx=0)`` per categorical channel and
have padded tokens lookup-to-zero automatically.

Numeric params are normalized to ~unit scale (``/_STAT_NORM`` for atk/hp,
``/_PARAM_NORM`` for amount/repeats/count) so the per-ability MLP doesn't see
order-of-magnitude differences across effect classes.

Pending discover/triple-reward options expand into ability tokens (we
materialize the chosen card via :func:`make_minion` and read its abilities);
adapt options have no minion and therefore stay zero-padded (the model already
sees the adapt key in the existing pending block).

Hand/shop/own minions that don't carry K abilities are padded with zeros — the
model masks them via ``effect_id == 0`` in :class:`BGLikeStructuredV5`.
"""

from __future__ import annotations

from typing import List, Mapping, Optional, Sequence

import numpy as np

from src.bg_catalog.cards import make_minion
from src.bg_catalog.patch_context import PatchContext
from src.bg_core.effects import Ability, ConditionKind, Keyword, Trigger
from src.bg_core.minion import Minion, Race
from src.bg_lobby.player import PendingChoice, PendingChoiceKind, PlayerState
from src.envs.minibg.obs import (
    EFFECT_INDEX,
    NUM_EFFECT_CHANNELS,
    NUM_TRIGGER_CHANNELS,
    RACE_ONEHOT_DIM,
    TRIGGER_INDEX,
    _RACE_ORDER,
)

from .actions import BOARD_SIZE, HAND_SIZE, MAX_SHOP_SLOTS
from .obs import OBS_DIM, build_observation
from .state import BGLikeState

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

K_ABIL = 4  # max abilities exposed per minion / pending option
PENDING_LEN = 3  # mirrors structured_common._PENDING_LEN

# Channel layout (all float32 in the obs vector; the model casts id channels
# back to long for embedding lookups):
#   [0..4]  5 base categorical ids: effect, trigger, cond_kind,
#           cond_arg_race, filter_race (Ability-level race filter)
#   [5]     combat_only flag
#   [6..10] 5 numeric params: atk, hp, amount, repeats, count
#   [11..14] 4 EXTENDED categorical ids appended at the tail so the base
#           offsets stay byte-identical: effect-level target tribe, the
#           effect's granted/filtered keyword, the Ability-level
#           filter_victim_keyword, and the summoned token's dense card index.
# All categorical ids carry the +1 shift (0 = pad / "no value").
_ABIL_ID_CHANNELS = 5
_ABIL_FLAG_CHANNELS = 1  # combat_only
_ABIL_NUMERIC_CHANNELS = 5  # atk, hp, amount, repeats, count
_ABIL_EXT_ID_CHANNELS = 4  # effect_tribe, effect_keyword, filter_victim_kw, summon_token
ABIL_FEAT_DIM = (
    _ABIL_ID_CHANNELS
    + _ABIL_FLAG_CHANNELS
    + _ABIL_NUMERIC_CHANNELS
    + _ABIL_EXT_ID_CHANNELS
)  # 15

# Field offsets inside one ability-feature vector
ABIL_OFF_EFFECT = 0
ABIL_OFF_TRIGGER = 1
ABIL_OFF_COND_KIND = 2
ABIL_OFF_COND_ARG_RACE = 3
ABIL_OFF_FILTER_RACE = 4
ABIL_OFF_COMBAT_ONLY = 5
ABIL_OFF_ATK = 6
ABIL_OFF_HP = 7
ABIL_OFF_AMOUNT = 8
ABIL_OFF_REPEATS = 9
ABIL_OFF_COUNT = 10
# Extended categorical ids (effect-level args the base channels miss).
ABIL_OFF_EFFECT_TRIBE = 11  # target tribe of tribal effects (race vocab)
ABIL_OFF_EFFECT_KEYWORD = 12  # keyword granted/required by the effect (kw vocab)
ABIL_OFF_FILTER_VICTIM_KW = 13  # Ability.filter_victim_keyword (kw vocab)
ABIL_OFF_SUMMON_TOKEN = 14  # dense card idx of the summoned token (card vocab)

# Vocab sizes (incl. the padding/none id at index 0). +1 because the
# in-obs id is stored as ``raw_index + 1`` so 0 stays free for padding.
NUM_EFFECT_IDS = NUM_EFFECT_CHANNELS + 1
NUM_TRIGGER_IDS = NUM_TRIGGER_CHANNELS + 1
NUM_CONDITION_KIND_IDS = len(ConditionKind) + 1
NUM_RACE_IDS = RACE_ONEHOT_DIM + 1  # _RACE_ORDER already includes None at 0
NUM_KEYWORD_IDS = len(Keyword) + 1  # Keyword.value is 1..N (auto()), 0 = pad

# Region token counts (kept aligned with the structured net layout)
_REGION_TOKEN_COUNTS = {
    "own": BOARD_SIZE,
    "shop": MAX_SHOP_SLOTS,
    "hand": HAND_SIZE,
    "pending": PENDING_LEN,
}
_NUM_ABIL_SLOT_TOKENS = (
    _REGION_TOKEN_COUNTS["own"]
    + _REGION_TOKEN_COUNTS["shop"]
    + _REGION_TOKEN_COUNTS["hand"]
    + _REGION_TOKEN_COUNTS["pending"]
)
ABIL_BLOCK_DIM = _NUM_ABIL_SLOT_TOKENS * K_ABIL * ABIL_FEAT_DIM

OBS_DIM_V5 = OBS_DIM + ABIL_BLOCK_DIM

# Per-region offsets inside the ability block (each region is a flat
# (slots * K_ABIL * ABIL_FEAT_DIM) chunk in own/shop/hand/pending order).
ABIL_BLOCK_OFFSET = OBS_DIM
ABIL_OWN_OFFSET = ABIL_BLOCK_OFFSET
ABIL_SHOP_OFFSET = ABIL_OWN_OFFSET + _REGION_TOKEN_COUNTS["own"] * K_ABIL * ABIL_FEAT_DIM
ABIL_HAND_OFFSET = ABIL_SHOP_OFFSET + _REGION_TOKEN_COUNTS["shop"] * K_ABIL * ABIL_FEAT_DIM
ABIL_PENDING_OFFSET = ABIL_HAND_OFFSET + _REGION_TOKEN_COUNTS["hand"] * K_ABIL * ABIL_FEAT_DIM

# Normalization scales — keep effect-class-agnostic so attention can compare.
_STAT_NORM = 5.0  # atk/hp
_PARAM_NORM = 5.0  # amount/repeats/count


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------


def _race_id(race: Optional[Race]) -> int:
    """Position in ``_RACE_ORDER`` (0 = None). Shifted +1 for pad."""
    try:
        return _RACE_ORDER.index(race) + 1
    except ValueError:
        return 0 + 1  # treat unknown races as 'None'


def _effect_id(effect_obj) -> int:
    """``EFFECT_INDEX[type(eff)] + 1`` (1-based; 0 reserved for padding).

    Raises if the effect class isn't registered — an unregistered effect would
    collapse to the padding id and be silently masked out of attention, so we
    fail loudly instead. ``_EFFECT_CLASSES`` is kept complete by the import-time
    guard in ``minibg.obs``."""
    idx = EFFECT_INDEX.get(type(effect_obj))
    if idx is None:
        raise KeyError(
            f"effect {type(effect_obj).__name__} not in EFFECT_INDEX; "
            "add it to _EFFECT_CLASSES (minibg.obs)."
        )
    return idx + 1


def _trigger_id(trigger: Trigger) -> int:
    idx = TRIGGER_INDEX.get(trigger)
    if idx is None:
        raise KeyError(
            f"trigger {trigger!r} not in TRIGGER_INDEX; add it to the registry."
        )
    return idx + 1


def _condition_kind_id(kind: Optional[ConditionKind]) -> int:
    if kind is None:
        return 0
    return int(kind.value)  # enum auto() starts at 1 → fits "0 = pad" already


def _keyword_id(kw: Optional[Keyword]) -> int:
    """``Keyword.value`` (auto() → 1..N) or 0 if absent. 0 = pad."""
    if kw is None:
        return 0
    try:
        return int(kw.value)
    except (TypeError, ValueError):
        return 0


# Effect-level race/tribe is stored under different attribute names across the
# ~30 tribal effect classes. Probe in priority order: the explicit tribal
# target first, then race filters, then the first of a tribe tuple.
_EFFECT_TRIBE_ATTRS = ("tribe", "race_filter", "filter_race", "race")


def _effect_tribe_id(eff) -> int:
    """Race id of the effect's target/filter tribe (0 if the effect has none).

    Captures the categorical arg that the base ``filter_race`` (Ability-level)
    and ``condition`` channels miss — e.g. ``BuffAllFriendlyOfTribe(tribe=MURLOC)``
    stores its target tribe on the *effect*, so without this the model can't
    tell "+2/+0 to Murlocs" from "+2/+0 to Beasts"."""
    for attr in _EFFECT_TRIBE_ATTRS:
        val = getattr(eff, attr, None)
        if val is not None:
            return _race_id(val)
    tribes = getattr(eff, "tribes", None)
    if tribes:
        return _race_id(tribes[0])
    return 0


def _summon_token_idx(eff, patch: PatchContext) -> int:
    """Dense card index of the fixed token an effect summons (0 if none / not a
    known template). Routed through the shared ``card_emb`` model-side so a
    summoned 'Murloc Scout' shares its representation with the shop one."""
    tok = getattr(eff, "token_id", None)
    if tok is None:
        return 0
    try:
        return int(patch.card_id_to_dense.get(tok, 0))
    except Exception:
        return 0


def _numeric_param(eff, name: str) -> float:
    """Read ``eff.name`` if present, normalize. Missing → 0."""
    if not hasattr(eff, name):
        return 0.0
    try:
        val = float(getattr(eff, name))
    except (TypeError, ValueError):
        return 0.0
    if name in ("attack", "health"):
        return val / _STAT_NORM
    return val / _PARAM_NORM


# ---------------------------------------------------------------------------
# Per-ability / per-minion encoders
# ---------------------------------------------------------------------------


def encode_ability_token(ab: Ability, patch: PatchContext) -> np.ndarray:
    """Pack one ``Ability`` into an ``ABIL_FEAT_DIM``-float vector.

    Order matches the ``ABIL_OFF_*`` offsets above. All ids carry the +1
    pad-shift; padded tokens have ``effect_id == 0`` and the embedding
    table's ``padding_idx=0`` returns zeros automatically.
    """
    v = np.zeros(ABIL_FEAT_DIM, dtype=np.float32)
    eff = ab.effect
    v[ABIL_OFF_EFFECT] = float(_effect_id(eff))
    v[ABIL_OFF_TRIGGER] = float(_trigger_id(ab.trigger))
    cond = ab.condition
    if cond is not None:
        v[ABIL_OFF_COND_KIND] = float(_condition_kind_id(cond.kind))
        v[ABIL_OFF_COND_ARG_RACE] = float(_race_id(cond.tribe))
    v[ABIL_OFF_FILTER_RACE] = float(_race_id(ab.filter_race))
    v[ABIL_OFF_COMBAT_ONLY] = 1.0 if bool(ab.combat_only) else 0.0
    v[ABIL_OFF_ATK] = _numeric_param(eff, "attack")
    v[ABIL_OFF_HP] = _numeric_param(eff, "health")
    v[ABIL_OFF_AMOUNT] = _numeric_param(eff, "amount")
    v[ABIL_OFF_REPEATS] = _numeric_param(eff, "repeats")
    v[ABIL_OFF_COUNT] = _numeric_param(eff, "count")
    # Extended categorical args (effect-level tribe / keyword / summoned token,
    # plus the Ability-level victim-keyword filter).
    v[ABIL_OFF_EFFECT_TRIBE] = float(_effect_tribe_id(eff))
    v[ABIL_OFF_EFFECT_KEYWORD] = float(_keyword_id(getattr(eff, "keyword", None)))
    v[ABIL_OFF_FILTER_VICTIM_KW] = float(_keyword_id(ab.filter_victim_keyword))
    v[ABIL_OFF_SUMMON_TOKEN] = float(_summon_token_idx(eff, patch))
    return v


def encode_minion_abilities(
    minion: Optional[Minion], patch: PatchContext
) -> np.ndarray:
    """``(K_ABIL, ABIL_FEAT_DIM)`` — first ``K_ABIL`` abilities, zero-padded."""
    out = np.zeros((K_ABIL, ABIL_FEAT_DIM), dtype=np.float32)
    if minion is None:
        return out
    for i, ab in enumerate(minion.abilities[:K_ABIL]):
        out[i] = encode_ability_token(ab, patch)
    return out


def encode_pending_option_abilities(
    pc: Optional[PendingChoice], patch: PatchContext
) -> np.ndarray:
    """``(PENDING_LEN, K_ABIL, ABIL_FEAT_DIM)`` for the 3 pending options.

    Discover / triple-reward options carry a ``card_id`` — we instantiate the
    minion and read its abilities. ADAPT / APPLY_EFFECT / TRANSFORM_SHOP_MINION
    options aren't minion-typed, so they stay zero-padded (the existing pending
    block tells the model what kind of modal is open).
    """
    out = np.zeros((PENDING_LEN, K_ABIL, ABIL_FEAT_DIM), dtype=np.float32)
    if pc is None:
        return out
    # Only minion-style discover options yield meaningful ability tokens.
    if pc.kind not in (
        PendingChoiceKind.DISCOVER_MURLOC,
        PendingChoiceKind.TRIPLE_REWARD_DISCOVER,
    ):
        return out
    for i, card_id in enumerate(pc.options[:PENDING_LEN]):
        try:
            m = make_minion(card_id, patch=patch)
        except Exception:
            # Some pending kinds may carry non-card tokens; keep them padded.
            continue
        out[i] = encode_minion_abilities(m, patch)
    return out


def _encode_minion_seq_abilities(
    minions: Sequence[Optional[Minion]], num_slots: int, patch: PatchContext
) -> np.ndarray:
    out = np.zeros((num_slots, K_ABIL, ABIL_FEAT_DIM), dtype=np.float32)
    for i, m in enumerate(minions[:num_slots]):
        if m is None:
            continue
        out[i] = encode_minion_abilities(m, patch)
    return out


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


def build_observation_v5(
    state: BGLikeState,
    seat: int,
    last_battle_signed: float,
    *,
    is_my_turn: bool,
    patch: PatchContext,
    rl_pending=None,
) -> np.ndarray:
    """Return v5 obs (existing bglike obs + ability block at tail)."""
    base = build_observation(
        state,
        seat,
        last_battle_signed,
        is_my_turn=is_my_turn,
        patch=patch,
        rl_pending=rl_pending,
    )

    me = state.players[seat]

    own_abil = _encode_minion_seq_abilities(me.board, BOARD_SIZE, patch)
    shop_abil = _encode_minion_seq_abilities(me.shop, MAX_SHOP_SLOTS, patch)
    hand_abil = _encode_minion_seq_abilities(me.hand, HAND_SIZE, patch)
    pending_abil = encode_pending_option_abilities(me.pending_choice, patch)

    return np.concatenate(
        [
            base,
            own_abil.reshape(-1),
            shop_abil.reshape(-1),
            hand_abil.reshape(-1),
            pending_abil.reshape(-1),
        ]
    )


__all__ = [
    "K_ABIL",
    "ABIL_FEAT_DIM",
    "ABIL_BLOCK_DIM",
    "ABIL_BLOCK_OFFSET",
    "ABIL_OWN_OFFSET",
    "ABIL_SHOP_OFFSET",
    "ABIL_HAND_OFFSET",
    "ABIL_PENDING_OFFSET",
    "OBS_DIM_V5",
    "PENDING_LEN",
    "NUM_EFFECT_IDS",
    "NUM_TRIGGER_IDS",
    "NUM_CONDITION_KIND_IDS",
    "NUM_RACE_IDS",
    "NUM_KEYWORD_IDS",
    "ABIL_OFF_EFFECT",
    "ABIL_OFF_TRIGGER",
    "ABIL_OFF_COND_KIND",
    "ABIL_OFF_COND_ARG_RACE",
    "ABIL_OFF_FILTER_RACE",
    "ABIL_OFF_COMBAT_ONLY",
    "ABIL_OFF_ATK",
    "ABIL_OFF_HP",
    "ABIL_OFF_AMOUNT",
    "ABIL_OFF_REPEATS",
    "ABIL_OFF_COUNT",
    "ABIL_OFF_EFFECT_TRIBE",
    "ABIL_OFF_EFFECT_KEYWORD",
    "ABIL_OFF_FILTER_VICTIM_KW",
    "ABIL_OFF_SUMMON_TOKEN",
    "encode_ability_token",
    "encode_minion_abilities",
    "encode_pending_option_abilities",
    "build_observation_v5",
]
