"""Effects written twice must not lose a detail in one of the copies.

Two bugs of that exact shape, found by auditing every effect that is
implemented in both the shop and combat:

* Khadgar's summon multiplier was consulted only in combat, so a battlecry
  summon in the tavern was never doubled.
* ``DealDamageRandomEnemyMinion`` is implemented three times and the
  deathrattle copy never read ``repeats``, so golden Kaboom Bot dealt its
  damage once instead of twice.

Neither was a logic error — each was a second copy that knew less than the
first. These pin the behaviour; the parity checks at the bottom pin the
*shape*, so a future copy cannot quietly drift again.
"""

from __future__ import annotations

import numpy as np
import pytest

import src.envs.minibg  # noqa: F401  (breaks a circular import at collection)
from src.bg_catalog.cards import make_minion
from src.bg_catalog.patch_context import load_patch_context
from src.bg_combat.battle import simulate_battle
from src.bg_core.effects import (
    Ability,
    DealDamageRandomEnemyMinion,
    MultiplierKind,
    Trigger,
)
from src.bg_lobby.player import PlayerState
from src.bg_recruitment.place import place_from_hand
from src.bg_recruitment.shop_triggers import ShopTriggers

PATCH = "data/bgcore/19_6_0_74257"
KHADGAR = "DAL_575"
TIDEHUNTER = "EX1_506"  # Battlecry: summon a 1/1 Murloc Scout
SCOUT = "EX1_506a"


@pytest.fixture(scope="module")
def ctx():
    return load_patch_context(PATCH)


def _play_tidehunter(ctx, board_extra):
    triggers = ShopTriggers(np.random.default_rng(0), patch=ctx)
    player = PlayerState(
        health=40,
        gold=10,
        tavern_tier=3,
        next_tier_up_cost=5,
        board=list(board_extra),
        shop=[None] * 6,
        hand=[make_minion(TIDEHUNTER, patch=ctx)] + [None] * 4,
        phase=0,
        shop_actions_used=0,
    )
    place_from_hand(
        player, 0, None, board_size=7, triggers=triggers, rng=np.random.default_rng(0)
    )
    return sum(1 for m in player.board if m.card_id == SCOUT)


def test_khadgar_doubles_a_shop_summon(ctx):
    """'Your cards that summon minions summon twice as many' — no phase clause."""
    assert _play_tidehunter(ctx, []) == 1
    assert _play_tidehunter(ctx, [make_minion(KHADGAR, patch=ctx)]) == 2


def test_shop_summon_multiplier_matches_the_combat_one(ctx):
    """Both phases must read the same aura the same way."""
    from src.bg_combat.battle.auras import _summon_multiplier

    board = [make_minion(KHADGAR, patch=ctx), make_minion(KHADGAR, patch=ctx)]
    assert ShopTriggers.summon_multiplier(board) == 4  # product, not sum

    # A battle-side minion is just a Minion now, so the combat helper can be
    # handed the same board the shop helper got.
    class _Side:
        def __init__(self, minions):
            self.minions = minions

    for m in board:
        m.damage_taken = 0
    assert _summon_multiplier(_Side(board)) == 4
    assert ShopTriggers.summon_multiplier([]) == 1


def _enemy_deaths(ctx, repeats, seed):
    bomb = make_minion("BOT_606", patch=ctx)  # Kaboom Bot
    bomb.abilities = (
        Ability(Trigger.ON_DEATH, DealDamageRandomEnemyMinion(amount=4, repeats=repeats)),
    )
    bomb.base_attack, bomb.base_health = 0, 1  # dies immediately, kills nobody itself
    foes = []
    for _ in range(4):
        f = make_minion(TIDEHUNTER, patch=ctx)
        f.base_attack, f.base_health = 0, 4  # exactly one Kaboom hit each
        foes.append(f)
    foes[0].base_attack = 9  # kills the bomb
    log: list = []
    simulate_battle(
        [bomb],
        foes,
        p0_has_initiative=False,
        rng=np.random.default_rng(seed),
        combat_board_max=7,
        damage_cap=15,
        max_board_slots=7,
        patch=ctx,
        death_log=log,
    )
    return sum(1 for side, _cid in log if side == 1)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_deathrattle_damage_honours_repeats(ctx, seed):
    """Golden Kaboom Bot has repeats=2 and must hit twice."""
    assert _enemy_deaths(ctx, 1, seed) == 1
    assert _enemy_deaths(ctx, 2, seed) == 2


def test_golden_kaboom_bot_really_carries_repeats_2(ctx):
    """The catalog fact the fix exists for."""
    golden = ctx.triple_merge_golden_abilities("BOT_606")
    on_death = [a for a in golden if a.trigger is Trigger.ON_DEATH]
    assert len(on_death) == 1
    assert isinstance(on_death[0].effect, DealDamageRandomEnemyMinion)
    assert on_death[0].effect.repeats == 2


def test_summon_multiplier_aura_is_consulted_in_both_phases():
    """Shape check: the aura's scope is both phases, so both must consult it.

    Khadgar's helper was called seven times in ``bg_combat`` and never in
    ``bg_recruitment``; that asymmetry *was* the bug. Baron (deathrattle) and
    Brann (battlecry) are legitimately single-phase and are not checked here.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    phases = {
        "shop": ["src/bg_recruitment"],
        "combat": ["src/bg_combat"],
    }
    seen = {}
    for phase, roots in phases.items():
        found = False
        for r in roots:
            for path in (root / r).rglob("*.py"):
                for node in ast.walk(ast.parse(path.read_text())):
                    if isinstance(node, ast.Attribute) and node.attr == "SUMMON":
                        found = True
        seen[phase] = found
    assert seen["shop"], "shop never references MultiplierKind.SUMMON"
    assert seen["combat"], "combat never references MultiplierKind.SUMMON"
