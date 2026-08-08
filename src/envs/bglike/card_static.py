"""Static per-card table: frozen rules-text embedding ⊕ ability magnitudes.

Everything a *card template* determines is static — its rules text, its
abilities' numeric parameters, whether it is the golden copy. None of that
depends on the runtime state, so none of it belongs in the observation: the
obs already carries a dense ``card_idx`` per slot (``CARD_IDX_OFFSET``), and
the net gathers a row from this table instead of the learned ``card_emb``.

That is the whole point of the v6 obs: the 1560-float per-ability tail of
``obs_v5`` disappears, because it re-encoded static facts on every single
observation.

Row layout (``ROW_DIM`` floats)::

    [ text embedding      : TEXT_DIM  ]   frozen, from build_card_text_table.py
    [ ability magnitudes  : NUM_DIM   ]   computed here from the patch DSL

Indexing is by ``(card_idx, is_golden)``::

    row = card_idx + (num_pool_indices + 1) * int(is_golden)

Golden matters: a golden minion keeps its normal ``card_id`` (see
``triples.make_forged_golden_minion``) but its abilities come from
``patch.triple_merge_golden_abilities`` and can differ in magnitude. Keying
on the pair covers ~6 of the ~10 percentage points of runtime minions whose
abilities diverge from their template. The remainder is magnetised mechs,
whose abilities are merged in from *another* card at runtime
(``merge_magnetic_inplace``); a template-keyed table cannot see those, and
:func:`magnetic_divergence_note` documents that this is deliberate.

Text carries the *mechanic template* only: the build script masks digits, so
"Summon a 1/1 Cat" and "Summon a 2/1 Cat" share one vector and the magnitudes
come from the numeric block below. Text carries no tavern tier at all (it is
not in the card text), so tier stays an explicit per-slot channel.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from src.bg_catalog.patch_context import PatchContext
from src.bg_core.effects import Ability

# --------------------------------------------------------------------------
# Ability magnitudes
# --------------------------------------------------------------------------

# Max abilities on a *card template* in this patch (0 on 39, 1 on 116, 2 on 4).
# Asserted in :func:`build_number_table` so a future patch cannot silently
# truncate. Runtime minions can exceed this only via magnetise, which this
# table does not model by construction (see the module docstring).
K_STATIC_ABIL = 2

# Numeric fields read off each ability's effect dataclass, in row order.
# Plain linear scalars — each divided by its own constant so the net sees a
# ~unit range without a shared divisor smearing unrelated quantities.
#
# ``exact_tier`` is a *filter* ("summon a minion of exactly tier 2"), not a
# magnitude; it lives here because it is an integer the DSL carries, and the
# alternative (a categorical channel) would reintroduce the id-shaped encoding
# this obs version exists to test against.
NUMBER_FIELDS: Tuple[str, ...] = (
    "attack",
    "health",
    "amount",
    "repeats",
    "count",
    "factor",
    "attack_per",
    "health_per",
    "gold_reward",
    "exact_tier",
)
NUMBER_NORMS: Tuple[float, ...] = (5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 6.0)
assert len(NUMBER_FIELDS) == len(NUMBER_NORMS)

# Per ability: a present flag + the numeric fields.
ABIL_NUM_DIM = 1 + len(NUMBER_FIELDS)
# Whole block: K abilities + a normalised "how many abilities" scalar.
NUM_DIM = K_STATIC_ABIL * ABIL_NUM_DIM + 1

_NORMS = np.asarray(NUMBER_NORMS, dtype=np.float32)


def encode_ability_numbers(abilities: Sequence[Ability]) -> np.ndarray:
    """``(NUM_DIM,)`` magnitudes for one card's abilities.

    Fields absent from an effect dataclass read as 0 — that is correct rather
    than lossy for this block, because "does this effect have a factor" is
    itself the signal (only the three multiplier auras and Glyph Guardian do).
    """
    out = np.zeros(NUM_DIM, dtype=np.float32)
    n = min(len(abilities), K_STATIC_ABIL)
    for i in range(n):
        eff = abilities[i].effect
        base = i * ABIL_NUM_DIM
        out[base] = 1.0
        for j, name in enumerate(NUMBER_FIELDS):
            v = getattr(eff, name, None)
            # bool is an int subclass; flags are not magnitudes.
            if v is None or isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            out[base + 1 + j] = float(v) / _NORMS[j]
    out[-1] = float(len(abilities)) / float(K_STATIC_ABIL)
    return out


def build_number_table(patch: PatchContext) -> np.ndarray:
    """``(2, num_pool_indices + 1, NUM_DIM)`` — [normal, golden] × dense index.

    Index 0 of each half is the padding row (empty slot) and stays zero.
    """
    n_rows = int(patch.num_pool_indices) + 1
    out = np.zeros((2, n_rows, NUM_DIM), dtype=np.float32)
    over: list[str] = []
    for i, card_id in enumerate(patch.card_index_ids):
        dense = i + 1
        normal = tuple(patch.templates[card_id].abilities or ())
        if len(normal) > K_STATIC_ABIL:
            over.append(f"{card_id} (normal, {len(normal)})")
        out[0, dense] = encode_ability_numbers(normal)

        try:
            golden = tuple(patch.triple_merge_golden_abilities(card_id))
        except Exception:
            golden = normal
        if len(golden) > K_STATIC_ABIL:
            over.append(f"{card_id} (golden, {len(golden)})")
        out[1, dense] = encode_ability_numbers(golden)

    if over:
        raise AssertionError(
            "K_STATIC_ABIL=%d truncates card templates in this patch: %s. "
            "Bump K_STATIC_ABIL (changes ROW_DIM → new obs/network version)."
            % (K_STATIC_ABIL, ", ".join(over))
        )
    return out


def magnetic_divergence_note() -> str:
    """Why a template-keyed table is knowingly blind to magnetised mechs."""
    return (
        "merge_magnetic_inplace concatenates the magnet's abilities onto the "
        "target at runtime, so a magnetised mech's abilities are not those of "
        "its card_id. Measured on self-play: 9.9% of observed minions diverge "
        "from their template, 61% of that is golden (covered by the golden "
        "half of this table), leaving ~3.9% uncovered. Accepted for this obs "
        "version; revisit by tracking magnet provenance on the Minion."
    )


# --------------------------------------------------------------------------
# Text embeddings (built offline)
# --------------------------------------------------------------------------

TEXT_TABLE_FILENAME = "card_text_table.npz"


def load_text_table(patch: PatchContext) -> Tuple[np.ndarray, dict]:
    """``(num_pool_indices + 1, TEXT_DIM)`` frozen text rows + build metadata.

    The artifact is keyed by ``card_ids`` and re-indexed here through the
    patch's dense mapping, so a rebuild that reorders cards cannot silently
    shift rows.
    """
    path = patch.patch_dir / TEXT_TABLE_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; build it with "
            f"`python -m scripts.build_card_text_table --patch-dir {patch.patch_dir}`"
        )
    with np.load(path, allow_pickle=False) as z:
        card_ids = [str(c) for c in z["card_ids"]]
        emb = np.asarray(z["text_emb"], dtype=np.float32)
        meta = {
            "model": str(z["model"]),
            "text_dim": int(z["text_dim"]),
            "mask_numbers": bool(z["mask_numbers"]),
            "pca_dim": int(z["pca_dim"]),
        }

    by_id = {c: emb[i] for i, c in enumerate(card_ids)}
    n_rows = int(patch.num_pool_indices) + 1
    out = np.zeros((n_rows, emb.shape[1]), dtype=np.float32)
    missing = []
    for i, card_id in enumerate(patch.card_index_ids):
        row = by_id.get(card_id)
        if row is None:
            missing.append(card_id)
            continue
        out[i + 1] = row
    if missing:
        raise AssertionError(
            f"{TEXT_TABLE_FILENAME} is stale: no rows for {missing[:8]} "
            f"({len(missing)} cards). Rebuild it for this patch."
        )
    return out, meta


def random_text_table(n_rows: int, text_dim: int, *, seed: int) -> np.ndarray:
    """Control arm: same shape, same frozen-ness, no semantics.

    A run using this isolates "the *text* carries mechanics" from "restructuring
    the card features into a static table helped". Without it a win over the
    v11 baseline is unattributable.
    """
    rng = np.random.default_rng(seed)
    out = rng.standard_normal((n_rows, text_dim)).astype(np.float32)
    out /= np.linalg.norm(out, axis=1, keepdims=True).clip(min=1e-6)
    out[0] = 0.0  # padding row
    return out


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

TEXT_MODE_TEXT = "text"
TEXT_MODE_RANDOM = "random"
TEXT_MODES = (TEXT_MODE_TEXT, TEXT_MODE_RANDOM)


def build_card_static_table(
    patch: PatchContext,
    *,
    text_mode: str = TEXT_MODE_TEXT,
    text_dim: Optional[int] = None,
    random_seed: int = 0,
) -> Tuple[np.ndarray, dict]:
    """``(2 * (num_pool_indices + 1), TEXT_DIM + NUM_DIM)`` frozen table.

    Row index for a slot is ``card_idx + n_rows * is_golden``; row 0 (and its
    golden twin) is the padding row and is all-zero.
    """
    if text_mode not in TEXT_MODES:
        raise ValueError(f"text_mode={text_mode!r} not in {TEXT_MODES}")

    n_rows = int(patch.num_pool_indices) + 1
    if text_mode == TEXT_MODE_TEXT:
        text, meta = load_text_table(patch)
        if text_dim is not None and text.shape[1] != int(text_dim):
            raise ValueError(
                f"card_text_dim={text_dim} but the artifact has "
                f"{text.shape[1]}; rebuild with --pca-dim {text_dim}"
            )
    else:
        if text_dim is None:
            raise ValueError("text_mode='random' needs an explicit card_text_dim")
        text = random_text_table(n_rows, int(text_dim), seed=int(random_seed))
        meta = {
            "model": "random",
            "text_dim": int(text_dim),
            "mask_numbers": True,
            "pca_dim": int(text_dim),
        }

    numbers = build_number_table(patch)  # (2, n_rows, NUM_DIM)
    halves = [np.concatenate([text, numbers[g]], axis=1) for g in (0, 1)]
    table = np.concatenate(halves, axis=0).astype(np.float32)
    table[0] = 0.0
    table[n_rows] = 0.0
    meta = dict(meta)
    meta.update({"n_rows": n_rows, "num_dim": NUM_DIM, "row_dim": table.shape[1]})
    return table, meta


def static_row_index(card_idx: np.ndarray, is_golden: np.ndarray, n_rows: int) -> np.ndarray:
    """Numpy mirror of the model-side gather index (used by tests)."""
    return card_idx.astype(np.int64) + n_rows * is_golden.astype(np.int64)


__all__ = [
    "K_STATIC_ABIL",
    "NUMBER_FIELDS",
    "NUMBER_NORMS",
    "ABIL_NUM_DIM",
    "NUM_DIM",
    "TEXT_TABLE_FILENAME",
    "TEXT_MODE_TEXT",
    "TEXT_MODE_RANDOM",
    "TEXT_MODES",
    "encode_ability_numbers",
    "build_number_table",
    "load_text_table",
    "random_text_table",
    "build_card_static_table",
    "static_row_index",
    "magnetic_divergence_note",
]
