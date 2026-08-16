"""Per-patch observation/action **layout** (vocabularies and widths).

Card *content* is patch-scoped via :class:`~src.bg_catalog.patch_context.PatchContext`
and numeric *rules* via :mod:`src.bg_catalog.ruleset`; this module gives the same
treatment to the shapes the encoders and the action space are built from — how
many races exist, how many tavern tiers, how many shop slots.

Those numbers currently live as module-level constants in the observation and
action modules, identical for every patch. A modern client build widens all of
them (three more tribes, a seventh tier, a seventh shop slot), and a constant
cannot be widened for one package without moving it for the others.

``DEFAULT_LAYOUT`` reproduces today's constants exactly, so a patch package
without a ``"layout"`` block in ``meta.json`` — i.e. every package that exists
today — keeps the layout it already has. ``tests/test_patch_layout.py`` pins
that equality against the encoders themselves, and
``tests/test_patch_layout_snapshot.py`` pins the resulting widths per package.

Deliberately excluded: ``BOARD_SIZE`` / ``HAND_SIZE``. Both are real BG
constants that have never changed across builds, and both are baked into the
flat action-space layout; giving them a knob would suggest a freedom that does
not exist. ``max_shop_slots`` is here because it genuinely differs on modern
builds (a tier-7 tavern shows seven offers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence, Tuple

from src.bg_core.minion import Race

__all__ = [
    "PatchLayout",
    "DEFAULT_LAYOUT",
    "layout_from_meta",
    "LayoutValidationError",
]

# Mirrors src.envs.minibg.obs._RACE_ORDER — index 0 is "no race" (a neutral
# minion), and every other index is the one-hot position of that tribe. Order
# is load-bearing: it is the encoding, so a new patch appends, never inserts.
_RACES: Tuple[Race, ...] = (
    Race.BEAST,
    Race.DEMON,
    Race.MECHANICAL,
    Race.MURLOC,
    Race.DRAGON,
    Race.PIRATE,
    Race.ELEMENTAL,
    Race.ALL,
)

# Mirrors src.envs.bglike.actions.SHOP_OFFERS_BY_TIER.
_SHOP_OFFERS_BY_TIER: Mapping[int, int] = {1: 3, 2: 4, 3: 4, 4: 5, 5: 5, 6: 6}


class LayoutValidationError(ValueError):
    """A patch package's cards do not fit the layout it declares."""


@dataclass(frozen=True)
class PatchLayout:
    """Vocabulary sizes the observation encoders and action space are built on."""

    # Tribes in one-hot order, *excluding* the leading "no race" slot.
    races: Tuple[Race, ...] = field(default_factory=lambda: tuple(_RACES))
    # Tavern tiers a minion can have (one-hot width; tiers are 1..num_tiers).
    num_tiers: int = 6
    # Shop offer slots the action space reserves (the per-tier counts below
    # say how many are actually filled at each tavern tier).
    max_shop_slots: int = 6
    shop_offers_by_tier: Mapping[int, int] = field(
        default_factory=lambda: dict(_SHOP_OFFERS_BY_TIER)
    )
    # Options a Discover offers (three on every build so far; modern trinket
    # offers show four).
    discover_picks: int = 3

    @property
    def race_order(self) -> Tuple[Optional[Race], ...]:
        """One-hot order including the leading ``None`` ("no race") slot."""
        return (None,) + self.races

    @property
    def race_onehot_dim(self) -> int:
        return len(self.races) + 1

    def race_index(self, race: Optional[Race]) -> int:
        return self.race_order.index(race)

    def shop_offers(self, tavern_tier: int) -> int:
        return self.shop_offers_by_tier.get(int(tavern_tier), self.max_shop_slots)


DEFAULT_LAYOUT = PatchLayout()


def layout_from_meta(raw: Optional[Mapping]) -> PatchLayout:
    """Build a :class:`PatchLayout` from ``meta.json["layout"]`` (or ``None``).

    Every key is optional; a missing key falls back to ``DEFAULT_LAYOUT``'s
    field. ``races`` is a list of HS race names in one-hot order, without the
    leading "no race" slot.
    """
    if not raw:
        return DEFAULT_LAYOUT
    kwargs = {}
    if "races" in raw:
        kwargs["races"] = tuple(_race_from_name(name) for name in raw["races"])
    for key in ("num_tiers", "max_shop_slots", "discover_picks"):
        if key in raw:
            kwargs[key] = int(raw[key])
    if "shop_offers_by_tier" in raw:
        kwargs["shop_offers_by_tier"] = {
            int(k): int(v) for k, v in raw["shop_offers_by_tier"].items()
        }
    return PatchLayout(**kwargs)


def _race_from_name(name: str) -> Race:
    try:
        return Race[str(name).strip().upper()]
    except KeyError:
        raise LayoutValidationError(
            f"unknown race {name!r} in layout.races; known: {[r.name for r in Race]}"
        ) from None


def validate_layout(
    layout: PatchLayout,
    *,
    max_tier: int,
    rotation_tribes: Sequence[Race],
    card_races: Iterable[Optional[Race]],
    card_tiers: Iterable[int],
    package: str = "",
) -> None:
    """Reject a package whose cards or rules do not fit its declared layout.

    A race the encoder has no one-hot column for, or a tier past the end of the
    tier one-hot, would otherwise be encoded as *something else* — silently, on
    every observation. This is the check that makes a data-driven layout safe.
    """
    where = f" in {package}" if package else ""

    duplicates = [r.name for r in set(layout.races) if list(layout.races).count(r) > 1]
    if duplicates:
        raise LayoutValidationError(f"layout.races repeats {sorted(duplicates)}{where}")

    if layout.num_tiers < int(max_tier):
        raise LayoutValidationError(
            f"layout.num_tiers={layout.num_tiers} cannot encode ruleset.max_tier="
            f"{int(max_tier)}{where}"
        )

    known = set(layout.races)
    missing_rotation = sorted(r.name for r in rotation_tribes if r not in known)
    if missing_rotation:
        raise LayoutValidationError(
            f"rotation tribes {missing_rotation} are absent from layout.races{where}"
        )

    missing_cards = sorted(
        {r.name for r in card_races if r is not None and r not in known}
    )
    if missing_cards:
        raise LayoutValidationError(
            f"catalog cards carry races {missing_cards} absent from layout.races{where}"
        )

    # Tier 0 means "no tavern tier" — the triple-reward spell and any token that
    # never appears in a tavern. The encoder writes an all-zero tier one-hot for
    # those, which is a faithful encoding, not a truncation. A tier *past* the
    # end of the one-hot is the real hazard: it silently reads as tier-less.
    over_tier = sorted({int(t) for t in card_tiers if not 0 <= int(t) <= layout.num_tiers})
    if over_tier:
        raise LayoutValidationError(
            f"catalog cards carry tiers {over_tier} outside 0..{layout.num_tiers}{where}"
        )

    for tier, count in sorted(layout.shop_offers_by_tier.items()):
        if count > layout.max_shop_slots:
            raise LayoutValidationError(
                f"shop_offers_by_tier[{tier}]={count} exceeds max_shop_slots="
                f"{layout.max_shop_slots}{where}"
            )
    uncovered = sorted(set(range(1, int(max_tier) + 1)) - set(layout.shop_offers_by_tier))
    if uncovered:
        raise LayoutValidationError(
            f"shop_offers_by_tier has no entry for tiers {uncovered}{where}"
        )
