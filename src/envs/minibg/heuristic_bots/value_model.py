"""Card-agnostic value estimates for Mini BG scripted play."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Optional, Sequence, Tuple

from ..actions import MAX_ROUNDS, MAX_TIER
from ..effects import (
    AdaptAllMurlocsEffect,
    AdjacentStatAura,
    AttackBonusPerOtherMurlocGlobal,
    BattlecryMultiplierAura,
    DeathrattleMultiplierAura,
    DiscoverMurlocEffect,
    HeroImmuneAura,
    KeywordStatAura,
    StatAura,
    SummonEffect,
    SummonMultiplierAura,
    SummonRandomMinionEffect,
    TribalOtherStatAura,
    Trigger,
    ZappTargeting,
)
from ..effects import Keyword
from ..state import Minion, Race


def rounds_left_estimate(round_number: int) -> int:
    return max(1, MAX_ROUNDS - int(round_number))


def tribe_counts(board: Sequence[Minion]) -> Dict[Race, int]:
    c: Counter[Race] = Counter()
    for m in board:
        if m.race is not None:
            c[m.race] += 1
    return dict(c)


def dominant_race(board: Sequence[Minion]) -> Optional[Race]:
    c = tribe_counts(board)
    if not c:
        return None
    return max(c.items(), key=lambda kv: kv[1])[0]


def ability_shop_estimate(m: Minion, rounds_left: int, board_len: int) -> float:
    bl = max(1, board_len)
    add = 0.0
    for ab in m.abilities:
        tr = ab.trigger
        eff = ab.effect
        if tr == Trigger.ON_DEATH:
            if isinstance(eff, SummonEffect):
                add += 3.6 * float(eff.count)
            elif isinstance(eff, SummonRandomMinionEffect):
                add += 4.2 * float(eff.count)
            else:
                add += 2.0
        elif tr == Trigger.ON_BUY:
            add += 2.5
        elif tr == Trigger.ON_PLACE:
            if isinstance(eff, DiscoverMurlocEffect):
                add += 6.5 * float(eff.repeats)
            elif isinstance(eff, AdaptAllMurlocsEffect):
                add += 9.0 * float(eff.repeats)
            else:
                add += 2.2
        elif tr == Trigger.AURA:
            if isinstance(eff, StatAura):
                add += (eff.attack + eff.health) * min(4, bl) * 0.38
            elif isinstance(eff, TribalOtherStatAura):
                add += (eff.attack + eff.health) * min(4, bl) * 0.28
            elif isinstance(eff, AdjacentStatAura):
                add += (eff.attack + eff.health) * 2.2
            elif isinstance(eff, KeywordStatAura):
                add += (eff.attack + eff.health) * 2.2
            elif isinstance(eff, BattlecryMultiplierAura):
                add += 5.5 * float(max(0, eff.factor - 1))
            elif isinstance(eff, DeathrattleMultiplierAura):
                add += 4.5 * float(max(0, eff.factor - 1))
            elif isinstance(eff, SummonMultiplierAura):
                add += 4.5 * float(max(0, eff.factor - 1))
            elif isinstance(eff, HeroImmuneAura):
                add += 14.0
            elif isinstance(eff, AttackBonusPerOtherMurlocGlobal):
                add += float(eff.per_attack) * 2.0
            elif isinstance(eff, ZappTargeting):
                add += 2.5
        elif tr == Trigger.ON_TURN_START:
            add += 2.0 * float(min(rounds_left, 10))
        elif tr == Trigger.ON_TURN_END:
            add += 1.6 * float(min(rounds_left, 10))
        elif tr == Trigger.AFTER_FRIENDLY_MINION_PLACED:
            add += 3.2
    return add


def minion_shop_value(
    m: Minion,
    *,
    rounds_left: int,
    dominant: Optional[Race],
    board_len: int,
    round_number: int = 8,
    tavern_tier_cap: Optional[int] = None,
) -> float:
    atk = float(m.raw_attack)
    hp = float(m.max_health)
    v = atk + hp + 0.055 * atk * hp
    rn = max(1, int(round_number))
    curve = max(0.0, min(1.0, (float(rn) - 4.0) / 13.0))
    v += float(min(m.tier, MAX_TIER)) * (0.38 + 0.62 * curve)
    if Keyword.TAUNT in m.all_keywords:
        v += 1.8
    if m.has_shield or Keyword.SHIELD in m.all_keywords:
        v += 3.0
    if Keyword.POISONOUS in m.all_keywords:
        v += 3.8
    if Keyword.WINDFURY in m.all_keywords:
        v += min(atk * 0.55, 5.0)
    if Keyword.CHARGE in m.all_keywords:
        v += 1.5
    if m.race is not None and dominant is not None and m.race == dominant:
        v *= 1.13
    if m.race == Race.ALL and dominant is not None:
        v *= 1.06
    if tavern_tier_cap is not None:
        cap = int(tavern_tier_cap)
        if m.tier >= cap:
            v += 3.4 + 0.22 * float(m.tier)
        elif m.tier == cap - 1 and cap > 1:
            v += 1.35
    v += ability_shop_estimate(m, rounds_left, board_len)
    return v


def board_power(
    board: Sequence[Minion],
    *,
    rounds_left: int,
    round_number: int = 8,
) -> float:
    if not board:
        return 0.0
    dom = dominant_race(board)
    bl = len(board)
    return sum(
        minion_shop_value(
            m,
            rounds_left=rounds_left,
            dominant=dom,
            board_len=bl,
            round_number=round_number,
        )
        for m in board
    )


def order_key_structured(
    old_idx: int, board: Sequence[Minion | None]
) -> Tuple[float, int]:
    m = board[old_idx]
    assert m is not None
    # Keys must not depend on ``old_idx`` (only on the minion). Shop reorder uses
    # adjacent swaps; if the target order changes every step, swaps need not
    # converge and the ORDER phase can run for thousands of steps.
    w = 0.0
    if Keyword.TAUNT in m.all_keywords:
        w -= 78.0
    if Keyword.CHARGE in m.all_keywords:
        w -= 12.0
    w -= float(m.raw_attack) * 2.25
    w -= float(m.max_health) * 0.28
    if Keyword.POISONOUS in m.all_keywords:
        w -= 28.0
    if any(ab.trigger == Trigger.ON_DEATH for ab in m.abilities):
        w += 20.0

    return (w, old_idx)


ADAPT_KEY_SCORE: Dict[str, float] = {
    "adapt_massive": 9.0,
    "adapt_volcanic_might": 2.8,
    "adapt_crackling_shield": 6.5,
    "adapt_flaming_claws": 4.5,
    "adapt_living_spore": 7.5,
    "adapt_lightning_speed": 5.0,
    "adapt_razor_claws": 2.2,
    "adapt_rocky_carapace": 4.0,
    "adapt_rockshell_armadillo": 8.5,
    "adapt_molten_blade": 3.8,
}


def adapt_choice_score(key: str) -> float:
    return ADAPT_KEY_SCORE.get(key, 1.0)


def roll_value_threshold(tavern_tier: int, round_number: int) -> float:
    return 3.95 + 0.52 * float(tavern_tier) + 0.035 * float(max(0, round_number))


def roll_threshold_adjusted(
    *,
    tavern_tier: int,
    round_number: int,
    mine_board_power: float,
    opp_board_power: float,
) -> float:
    """Higher threshold ⇒ roll less often (keep weak-ish shop); lower ⇒ roll more."""
    thr = roll_value_threshold(tavern_tier, round_number)
    if opp_board_power < 1e-6:
        return thr + 1.65
    ratio = mine_board_power / opp_board_power
    if ratio < 0.93:
        return thr - 1.35
    if ratio > 1.14:
        return thr + 1.55
    return thr


__all__ = [
    "ADAPT_KEY_SCORE",
    "ability_shop_estimate",
    "adapt_choice_score",
    "board_power",
    "dominant_race",
    "minion_shop_value",
    "order_key_structured",
    "roll_threshold_adjusted",
    "roll_value_threshold",
    "rounds_left_estimate",
    "tribe_counts",
]
