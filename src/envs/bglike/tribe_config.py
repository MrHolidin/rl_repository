"""Resolve ``forced_tribes`` / ``excluded_tribes`` game params into shop exclusion."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from src.bg_catalog.cards import normalize_shop_excluded_races
from src.bg_catalog.patch_catalog import race_from_hs_string
from src.bg_catalog.patch_context import PatchContext, load_patch_context
from src.bg_core.minion import Race

TribeSpec = Union[str, Race, Sequence[Union[str, Race]]]


def _parse_tribe_list(
    raw: TribeSpec,
    *,
    rotation: Tuple[Race, ...],
    label: str,
) -> Tuple[Race, ...]:
    if isinstance(raw, Race):
        items: Sequence[Union[str, Race]] = (raw,)
    elif isinstance(raw, str):
        items = (raw,)
    else:
        items = raw

    out: list[Race] = []
    seen: set[Race] = set()
    for item in items:
        if isinstance(item, Race):
            race = item
        else:
            race = race_from_hs_string(str(item).strip().upper())
        if race is Race.ALL:
            raise ValueError(f"{label}: Race.ALL is not allowed")
        if race not in rotation:
            names = ", ".join(r.name for r in rotation)
            raise ValueError(
                f"{label}: {race.name} is not in patch rotation ({names})"
            )
        if race in seen:
            continue
        out.append(race)
        seen.add(race)
    return tuple(out)


def resolve_shop_excluded_races(
    *,
    patch: PatchContext,
    forced_tribes: Optional[TribeSpec] = None,
    excluded_tribes: Optional[TribeSpec] = None,
) -> Optional[Tuple[Race, ...]]:
    """Map config tribe lists to ``shop_excluded_race`` for ``BGLikeGame``."""
    rotation = patch.meta.rotation_tribes
    if forced_tribes is None and excluded_tribes is None:
        return None

    if forced_tribes is not None and excluded_tribes is not None:
        raise ValueError("Specify forced_tribes or excluded_tribes, not both")

    if forced_tribes is not None:
        active = _parse_tribe_list(forced_tribes, rotation=rotation, label="forced_tribes")
        if not active:
            raise ValueError("forced_tribes must be non-empty")
        active_set = set(active)
        excluded = tuple(r for r in rotation if r not in active_set)
    else:
        excluded = _parse_tribe_list(
            excluded_tribes,  # type: ignore[arg-type]
            rotation=rotation,
            label="excluded_tribes",
        )

    if len(excluded) >= len(rotation):
        raise ValueError("At least one rotation tribe must remain in the shop pool")
    return excluded or None


def apply_tribe_params_to_lobby_kwargs(
    kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    """Resolve tribe aliases and strip non-constructor keys for lobby env factories."""
    out = dict(kwargs)
    forced = out.pop("forced_tribes", None)
    excluded = out.pop("excluded_tribes", None)
    if forced is None and excluded is None:
        return out

    if out.get("shop_full_tribes"):
        raise ValueError("shop_full_tribes conflicts with forced_tribes / excluded_tribes")

    patch_dir = out.get("patch_dir")
    if not patch_dir:
        raise ValueError(
            "forced_tribes / excluded_tribes require game.params.patch_dir"
        )
    patch = load_patch_context(str(patch_dir))
    resolved = resolve_shop_excluded_races(
        patch=patch,
        forced_tribes=forced,
        excluded_tribes=excluded,
    )

    legacy = out.get("shop_excluded_race")
    if legacy is not None:
        legacy_norm = normalize_shop_excluded_races(legacy)
        resolved_norm = normalize_shop_excluded_races(resolved)
        if legacy_norm != resolved_norm:
            raise ValueError(
                "shop_excluded_race conflicts with forced_tribes / excluded_tribes"
            )

    out["shop_excluded_race"] = resolved
    return out


__all__ = [
    "apply_tribe_params_to_lobby_kwargs",
    "resolve_shop_excluded_races",
]
