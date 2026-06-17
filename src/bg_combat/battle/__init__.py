"""Combat simulation, split into focused modules.

Public surface (and the private internals some tests / tools import) is
re-exported so ``from src.bg_combat.battle import X`` keeps working."""
from __future__ import annotations

from .events import (
    AttackCompleted,
    BattleEvent,
    BeginAttackExchange,
    DamageDealt,
    DamageStrike,
    MinionDied,
    MinionSummoned,
    Overkill,
    ShieldLost,
)
from .state import BattleMinion, BattleSide, _CombatRuntime
from .auras import (
    attack_value,
    attack_with_auras,
    health_aura_bonus,
    _sync_health_all,
    _sync_health_aura_side,
)
from .sides import build_battle_side, _build_side
from .targeting import _pick_target
from .summon import _summon_insert
from .effects import _deal_damage_to_battle_minion, _fire_deathrattle
from .engine import (
    _decide_first_side,
    _dispatch,
    _fire_start_of_combat,
    _next_attacker,
    _run_single_swing,
)
from .result import BattleResult, RawBattleSnapshot, persist_shop_board_from_side
from .simulate import simulate_battle

__all__ = [
    "BattleMinion",
    "BattleSide",
    "persist_shop_board_from_side",
    "BattleEvent",
    "BeginAttackExchange",
    "ShieldLost",
    "DamageDealt",
    "Overkill",
    "AttackCompleted",
    "MinionDied",
    "MinionSummoned",
    "DamageStrike",
    "attack_with_auras",
    "attack_value",
    "health_aura_bonus",
    "build_battle_side",
    "simulate_battle",
    "BattleResult",
    "RawBattleSnapshot",
]
