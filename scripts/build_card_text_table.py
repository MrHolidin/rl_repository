#!/usr/bin/env python3
"""Build the frozen card rules-text table consumed by ``card_static.py``.

Run once per patch. The output artifact is what training loads; ``transformers``
is a *build-time only* dependency and is never imported by the training path.

    python -m scripts.build_card_text_table \
        --patch-dir data/bgcore/19_6_0_74257

Digits are masked before encoding, on purpose. Measured on this patch, changing
one digit in a card's text moves the sentence embedding by cos 0.983 (min
0.956) while a different card sits at 0.380 — magnitudes occupy ~2% of the
encoder's dynamic range, so leaving them in adds noise to the only thing the
encoder does carry well (the mechanic template) and still fails to encode the
number. Magnitudes come from the DSL instead (``card_static.NUMBER_FIELDS``).

Masking is not free-floating speculation: with digits masked, same-mechanic
different-magnitude pairs move closer (Coldlight Seer / Felfin Navigator
0.887 → 0.923, mean +0.025 over a hand-picked set), the pairs the effect-id
encoding collapses stay just as separated (mean Δ −0.000), and the linear
probe for "which effect class" rises 0.545 → 0.585.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# 32 dims keep every linearly-readable mechanic: the "which trigger" probe
# scores 0.716 at both 32 and the full 384 (majority baseline 0.252), and 0.684
# at 16. Below 16 it degrades (0.598 at 8).
DEFAULT_PCA_DIM = 32

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_DIGIT_RE = re.compile(r"\d+")

# Engine-internal pseudo-templates: they hold a dense card index but are not
# Hearthstone cards, so there is no catalog row and no printed rules text. They
# get an all-zero text row; whatever mechanics they carry still reach the net
# through the ability-magnitude block and the per-slot channels. Listed
# explicitly so a genuinely missing *real* card still fails the build.
NO_CATALOG_ROW = frozenset({"adapt_plant", "target_buffer", "triple_reward_discover"})


def clean_text(raw: str | None, *, mask_numbers: bool) -> str:
    if not raw:
        return ""
    t = _TAG_RE.sub("", raw)
    t = t.replace("\\n", " ").replace("\n", " ").replace("[x]", "")
    t = _WS_RE.sub(" ", t).strip()
    if mask_numbers:
        t = _DIGIT_RE.sub("N", t)
    return t


def catalog_text_by_id(catalog_path: Path) -> Dict[str, str]:
    """card_id → raw rules text, preferring the non-golden row when both exist."""
    cat = json.loads(catalog_path.read_text())
    out: Dict[str, str] = {}
    for m in cat["minions"]:
        cid = m["id"]
        if cid in out and m.get("isGolden"):
            continue  # keep the non-golden row we already stored
        out[cid] = m.get("text") or ""
    return out


def embed_texts(texts: List[str], *, model_name: str, batch_size: int = 64) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).eval()

    chunks = []
    for i in range(0, len(texts), batch_size):
        batch = tok(
            texts[i : i + batch_size],
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        hidden = mdl(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        chunks.append(torch.nn.functional.normalize(pooled, dim=-1))
    return torch.cat(chunks).numpy().astype(np.float32)


def reduce_dim(emb: np.ndarray, dim: int, *, nonempty: np.ndarray) -> np.ndarray:
    """PCA down to ``dim``, fitted on cards that actually have text.

    Components are scaled to unit std so the frozen block enters the net at the
    same scale as the other per-slot channels. Cards without rules text keep an
    all-zero row (their mechanics live entirely in the slot channels).
    """
    if dim >= emb.shape[1]:
        return emb.copy()
    fit = emb[nonempty]
    mean = fit.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(fit - mean, full_matrices=False)
    comps = vt[:dim]
    out = (emb - mean) @ comps.T
    std = out[nonempty].std(axis=0, keepdims=True)
    out = out / np.clip(std, 1e-6, None)
    out[~nonempty] = 0.0
    return out.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patch-dir", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--pca-dim", type=int, default=DEFAULT_PCA_DIM)
    ap.add_argument(
        "--keep-numbers",
        action="store_true",
        help="do NOT mask digits (ablation; the default masks them)",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import src.envs  # noqa: F401  (resolves the bg package import cycle)
    from src.bg_catalog.patch_context import PatchContext
    from src.envs.bglike.card_static import TEXT_TABLE_FILENAME

    patch = PatchContext.load(args.patch_dir.resolve())
    raw_by_id = catalog_text_by_id(patch.patch_dir / "catalog.json")

    card_ids = list(patch.card_index_ids)
    mask_numbers = not args.keep_numbers
    texts = [clean_text(raw_by_id.get(c), mask_numbers=mask_numbers) for c in card_ids]
    nonempty = np.array([bool(t) for t in texts])
    missing = [c for c in card_ids if c not in raw_by_id and c not in NO_CATALOG_ROW]
    if missing:
        raise SystemExit(f"no catalog row for {len(missing)} templates: {missing[:8]}")
    print(
        f"{len(card_ids)} templates, {int(nonempty.sum())} with rules text "
        f"({len(NO_CATALOG_ROW)} engine-internal, {len(card_ids) - int(nonempty.sum()) - len(NO_CATALOG_ROW)} vanilla)"
    )

    full = embed_texts(texts, model_name=args.model)
    emb = reduce_dim(full, args.pca_dim, nonempty=nonempty)
    print(f"embedded {full.shape} -> reduced {emb.shape}")

    out_path = args.out or (patch.patch_dir / TEXT_TABLE_FILENAME)
    np.savez_compressed(
        out_path,
        card_ids=np.array(card_ids),
        text_emb=emb,
        text_emb_full=full,
        model=np.array(args.model),
        text_dim=np.array(emb.shape[1]),
        pca_dim=np.array(args.pca_dim),
        mask_numbers=np.array(mask_numbers),
    )
    print(f"wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
