"""Load a pinned BG patch package (catalog + meta + bindings) into runtime context."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Dict, FrozenSet, Mapping, Optional, Tuple

from src.bg_catalog.patch_catalog import (
    TavernMinionRecord,
    load_patch_catalog,
    load_tavern_minions,
    minion_by_id,
    minion_from_tavern_record,
    race_from_hs_string,
)
from src.bg_catalog.triple_effects import resolve_triple_forged_abilities
from src.bg_core.effects import Ability
from src.bg_core.minion import Minion, Race

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATCH_DIR = _REPO_ROOT / "data" / "bgcore" / "15_6_2_36393"


@dataclass(frozen=True)
class PatchMeta:
    rotation_tribes: Tuple[Race, ...]
    rotation_excluded_count: int
    pool_copies_by_tier: Mapping[int, int]

    @property
    def cnt_active_shop_tribes(self) -> int:
        return len(self.rotation_tribes) - self.rotation_excluded_count


@dataclass(frozen=True)
class PatchCardDescription:
    """Human-readable card entry for a patch (always includes display ``name``)."""

    card_id: str
    name: str
    template: Minion
    in_tavern_pool: bool
    catalog_text: Optional[str] = None


@dataclass(frozen=True)
class PatchContext:
    patch_dir: Path
    build: int
    patch: str
    templates: Mapping[str, Minion]
    descriptions: Mapping[str, PatchCardDescription]
    pool_ids: FrozenSet[str]
    meta: PatchMeta
    effects: Mapping[str, Tuple[Ability, ...]]
    golden_reward_ids: FrozenSet[str]
    token_ids: FrozenSet[str]
    keyword_only_pool_ids: FrozenSet[str]
    card_index_ids: Tuple[str, ...]
    card_id_to_dense: Mapping[str, int]
    num_pool_indices: int

    def make_minion(self, card_id: str) -> Minion:
        from copy import copy

        tpl = self.templates[card_id]
        fresh = copy(tpl)
        from src.bg_core.effects import Keyword

        fresh.has_shield = Keyword.SHIELD in tpl.all_keywords
        fresh.is_golden = tpl.is_golden
        fresh.from_triple_merge = False
        fresh.is_triple_reward_spell = tpl.is_triple_reward_spell
        fresh.triple_discover_tier = tpl.triple_discover_tier
        return fresh

    def triple_merge_golden_abilities(self, normal_card_id: str) -> Tuple[Ability, ...]:
        return resolve_triple_forged_abilities(normal_card_id, self.effects)

    def describe(self, card_id: str) -> PatchCardDescription:
        return self.descriptions[card_id]

    @classmethod
    def load(cls, patch_dir: Optional[Path] = None) -> PatchContext:
        root = (patch_dir or DEFAULT_PATCH_DIR).resolve()
        catalog_path = root / "catalog.json"
        meta_path = root / "meta.json"
        bindings_path = root / "bindings.py"
        for p in (catalog_path, meta_path, bindings_path):
            if not p.is_file():
                raise FileNotFoundError(f"patch package missing required file: {p}")

        catalog = load_patch_catalog(catalog_path)
        meta = _load_meta(meta_path)
        bindings = _load_bindings_module(bindings_path)
        effects: Dict[str, Tuple[Ability, ...]] = dict(bindings.EFFECTS)
        golden_reward_ids: FrozenSet[str] = frozenset(bindings.GOLDEN_REWARD_IDS)
        token_ids: FrozenSet[str] = frozenset(bindings.TOKEN_IDS)
        keyword_only_pool_ids: FrozenSet[str] = frozenset(
            getattr(bindings, "KEYWORD_ONLY_POOL_IDS", ())
        )

        templates, descriptions, pool_ids = _build_templates_and_descriptions(
            catalog_path=catalog_path,
            effects=effects,
            golden_reward_ids=golden_reward_ids,
            token_ids=token_ids,
        )
        card_index_ids, card_id_to_dense = _build_card_index(templates)
        return cls(
            patch_dir=root,
            build=int(catalog["build"]),
            patch=str(catalog["patch"]),
            templates=templates,
            descriptions=descriptions,
            pool_ids=pool_ids,
            meta=meta,
            effects=effects,
            golden_reward_ids=golden_reward_ids,
            token_ids=token_ids,
            keyword_only_pool_ids=keyword_only_pool_ids,
            card_index_ids=card_index_ids,
            card_id_to_dense=card_id_to_dense,
            num_pool_indices=len(card_index_ids),
        )


def _load_meta(path: Path) -> PatchMeta:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    tribes = tuple(race_from_hs_string(t) for t in raw["rotation_tribes"])
    copies = {int(k): int(v) for k, v in raw["pool_copies_by_tier"].items()}
    return PatchMeta(
        rotation_tribes=tribes,
        rotation_excluded_count=int(raw["rotation_excluded_count"]),
        pool_copies_by_tier=copies,
    )


def _load_bindings_module(path: Path) -> ModuleType:
    name = f"bg_patch_bindings_{path.parent.name}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load bindings module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _merge_template(
    rec_id: str,
    base: Minion,
    *,
    effects: Mapping[str, Tuple[Ability, ...]],
) -> Minion:
    fx = effects.get(rec_id, ())
    if not fx:
        return base
    return replace(base, abilities=fx)


def _description_for(
    rec: Optional[TavernMinionRecord],
    *,
    card_id: str,
    name: str,
    template: Minion,
    in_tavern_pool: bool,
    catalog_text: Optional[str] = None,
) -> PatchCardDescription:
    display_name = name or (rec.name if rec is not None else "") or card_id
    tpl = template if template.name else replace(template, name=display_name)
    return PatchCardDescription(
        card_id=card_id,
        name=display_name,
        template=tpl,
        in_tavern_pool=in_tavern_pool,
        catalog_text=rec.text if rec is not None else catalog_text,
    )


def _build_templates_and_descriptions(
    *,
    catalog_path: Path,
    effects: Mapping[str, Tuple[Ability, ...]],
    golden_reward_ids: FrozenSet[str],
    token_ids: FrozenSet[str],
) -> Tuple[Dict[str, Minion], Dict[str, PatchCardDescription], FrozenSet[str]]:
    rows = load_tavern_minions(catalog_path)
    by_id = minion_by_id(catalog_path)
    out: Dict[str, Minion] = {}
    descriptions: Dict[str, PatchCardDescription] = {}
    pool_ids: set[str] = set()

    for rec in rows:
        if rec.is_bacon_pool and not rec.is_golden:
            base = minion_from_tavern_record(rec)
            merged = _merge_template(rec.id, base, effects=effects)
            desc = _description_for(
                rec,
                card_id=rec.id,
                name=rec.name,
                template=merged,
                in_tavern_pool=True,
            )
            out[rec.id] = desc.template
            descriptions[rec.id] = desc
            pool_ids.add(rec.id)

    for gid in golden_reward_ids:
        rec = by_id[gid]
        base = minion_from_tavern_record(rec)
        merged = _merge_template(gid, base, effects=effects)
        desc = _description_for(
            rec,
            card_id=gid,
            name=rec.name,
            template=merged,
            in_tavern_pool=False,
        )
        out[gid] = desc.template
        descriptions[gid] = desc

    for tid in token_ids:
        rec = by_id[tid]
        base = minion_from_tavern_record(rec)
        merged = replace(
            _merge_template(tid, base, effects=effects),
            is_token=True,
        )
        desc = _description_for(
            rec,
            card_id=tid,
            name=rec.name,
            template=merged,
            in_tavern_pool=False,
        )
        out[tid] = desc.template
        descriptions[tid] = desc

    synthetic: Tuple[Tuple[str, str, Minion, bool], ...] = (
        (
            "adapt_plant",
            "Plant",
            Minion(
                card_id="adapt_plant",
                base_attack=2,
                base_health=2,
                tier=1,
                name="Plant",
                is_token=True,
            ),
            False,
        ),
        (
            "target_buffer",
            "Target Buffer",
            Minion(
                card_id="target_buffer",
                base_attack=2,
                base_health=2,
                tier=1,
                name="Target Buffer",
                abilities=effects["target_buffer"],
                is_token=True,
            ),
            False,
        ),
        (
            "triple_reward_discover",
            "Discover (Triple Reward)",
            Minion(
                card_id="triple_reward_discover",
                base_attack=0,
                base_health=0,
                tier=0,
                name="Discover (Triple Reward)",
                is_token=True,
                is_triple_reward_spell=True,
            ),
            False,
        ),
    )
    for card_id, name, base, in_pool in synthetic:
        desc = _description_for(
            None,
            card_id=card_id,
            name=name,
            template=base,
            in_tavern_pool=in_pool,
        )
        out[card_id] = desc.template
        descriptions[card_id] = desc

    return out, descriptions, frozenset(pool_ids)


def _build_card_index(
    templates: Mapping[str, Minion],
) -> Tuple[Tuple[str, ...], Dict[str, int]]:
    ids = tuple(sorted(templates.keys()))
    table = {cid: i + 1 for i, cid in enumerate(ids)}
    return ids, table


@lru_cache(maxsize=4)
def load_patch_context(patch_dir: Optional[str] = None) -> PatchContext:
    path = Path(patch_dir).resolve() if patch_dir is not None else DEFAULT_PATCH_DIR
    return PatchContext.load(path)


@lru_cache(maxsize=1)
def default_patch_context() -> PatchContext:
    """Pinned 36393 context for tests only — do not use as a runtime fallback."""
    return PatchContext.load(DEFAULT_PATCH_DIR)


def require_patch(patch: Optional[PatchContext], *, where: str) -> PatchContext:
    if patch is None:
        raise RuntimeError(f"patch is required ({where})")
    return patch


__all__ = [
    "DEFAULT_PATCH_DIR",
    "PatchCardDescription",
    "PatchContext",
    "PatchMeta",
    "default_patch_context",
    "load_patch_context",
    "require_patch",
]
