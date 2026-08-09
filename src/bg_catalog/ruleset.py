"""Per-patch numeric ruleset (tavern costs, gold curve, health, damage cap).

Card *content* is already patch-scoped via :class:`~src.bg_catalog.patch_context.PatchContext`
(``data/bgcore/<version>/``); this module gives the game's numeric *rules* the
same treatment, so a modern-patch package can carry different numbers without
disturbing older ones.

``DEFAULT_RULESET`` reproduces the constants that used to live in
``src.envs.bglike.actions`` / ``src.envs.minibg.actions`` exactly, so any
patch package without a ``"ruleset"`` block in ``meta.json`` (i.e. every
package that exists today) plays identically to before this module existed.

Deliberately excluded: ``BOARD_SIZE`` / ``HAND_SIZE`` / ``MAX_SHOP_SLOTS`` /
``SHOP_OFFERS_BY_TIER``. Those shape the flat RL action-space layout baked
into trained checkpoints and stay fixed architectural constants — making them
per-patch is a separate, larger "append-only masked action space" effort.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple

__all__ = [
    "DamageCapStep",
    "Ruleset",
    "DEFAULT_RULESET",
    "ruleset_from_meta",
]

# Mirrors src.envs.bglike.actions / src.envs.minibg.actions (identical in
# both today). Kept as private literals here so this module has no import
# dependency on the env layer (bg_catalog is a lower layer than envs).
_LEVEL_UP_COSTS: Mapping[int, int] = {1: 5, 2: 7, 3: 8, 4: 11, 5: 11}
_GOLD_PER_ROUND: Mapping[int, int] = {
    1: 3,
    2: 4,
    3: 5,
    4: 6,
    5: 7,
    6: 8,
    7: 9,
    8: 10,
}


@dataclass(frozen=True)
class DamageCapStep:
    """One entry of a damage-cap ramp: ``cap`` applies through ``max_round``
    inclusive. ``max_round=None`` means "this round onward", and must only
    appear as the last (highest) step."""

    max_round: Optional[int]
    cap: int


@dataclass(frozen=True)
class Ruleset:
    max_tier: int = 6
    level_up_costs: Mapping[int, int] = field(default_factory=lambda: dict(_LEVEL_UP_COSTS))
    level_up_discount_per_round: int = 1
    gold_per_round: Mapping[int, int] = field(default_factory=lambda: dict(_GOLD_PER_ROUND))
    gold_cap: int = 10
    buy_cost: int = 3
    sell_reward: int = 1
    roll_cost: int = 1
    starting_health: int = 40
    # Flat DAMAGE_CAP=15 forever — matches every existing patch package.
    damage_cap_schedule: Tuple[DamageCapStep, ...] = (DamageCapStep(None, 15),)
    # 0 ⇒ disabled: the cap never lifts (existing-patch behavior). A modern
    # patch sets this to 4 ("uncapped once ≤4 players remain alive").
    damage_cap_lifted_at_alive: int = 0
    max_rounds: int = 50

    def gold_for_round(self, round_number: int) -> int:
        return self.gold_per_round.get(int(round_number), self.gold_cap)

    def level_up_cost(self, tier: int) -> int:
        return self.level_up_costs.get(int(tier), 0)

    def damage_cap_for_round(self, round_number: int) -> int:
        rn = int(round_number)
        for step in self.damage_cap_schedule:
            if step.max_round is None or rn <= step.max_round:
                return step.cap
        # Unreachable if the schedule's last step has max_round=None, but
        # fall back to the last step's cap defensively.
        return self.damage_cap_schedule[-1].cap

    def effective_damage_cap(self, round_number: int, alive_count: int) -> int:
        """Damage cap for a combat happening this round with ``alive_count``
        players still alive. Lifted (effectively unbounded) once the field is
        down to ``damage_cap_lifted_at_alive`` or fewer, when that's enabled."""
        if self.damage_cap_lifted_at_alive and int(alive_count) <= self.damage_cap_lifted_at_alive:
            return 10**9
        return self.damage_cap_for_round(round_number)


DEFAULT_RULESET = Ruleset()


def ruleset_from_meta(raw: Optional[Mapping]) -> Ruleset:
    """Build a :class:`Ruleset` from ``meta.json["ruleset"]`` (or ``None``).

    Every key is optional; missing keys fall back to ``DEFAULT_RULESET``'s
    field. ``damage_cap_schedule`` entries are ``[max_round, cap]`` pairs in
    ascending ``max_round`` order (last entry's ``max_round`` may be
    ``null``/omitted for "onward").
    """
    if not raw:
        return DEFAULT_RULESET
    kwargs = {}
    for key in (
        "max_tier",
        "level_up_discount_per_round",
        "gold_cap",
        "buy_cost",
        "sell_reward",
        "roll_cost",
        "starting_health",
        "damage_cap_lifted_at_alive",
        "max_rounds",
    ):
        if key in raw:
            kwargs[key] = int(raw[key])
    if "level_up_costs" in raw:
        kwargs["level_up_costs"] = {int(k): int(v) for k, v in raw["level_up_costs"].items()}
    if "gold_per_round" in raw:
        kwargs["gold_per_round"] = {int(k): int(v) for k, v in raw["gold_per_round"].items()}
    if "damage_cap_schedule" in raw:
        kwargs["damage_cap_schedule"] = tuple(
            DamageCapStep(
                max_round=(int(entry[0]) if entry[0] is not None else None),
                cap=int(entry[1]),
            )
            for entry in raw["damage_cap_schedule"]
        )
    return Ruleset(**kwargs)
