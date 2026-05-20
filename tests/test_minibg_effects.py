from src.envs.minibg.actions import COMBAT_BOARD_MAX, DAMAGE_CAP, HAND_SIZE, LEVEL_UP_COSTS
from src.bg_catalog.cards import make_minion
from src.bg_combat.battle import (
    BattleMinion,
    BattleSide,
    _CombatRuntime,
    _fire_deathrattle,
    attack_with_auras,
)
from src.bg_core.effects import (
    Ability,
    Keyword,
    SummonFirstDeadFriendlyMechsThisCombat,
    Trigger,
)
from src.envs.minibg.state import PlayerPhase, PlayerState
from src.envs.minibg.game import MiniBGGame

import numpy as np


def _player(board=None, gold=10, tier=1):
    return PlayerState(
        health=20,
        gold=gold,
        tavern_tier=tier,
        next_tier_up_cost=LEVEL_UP_COSTS[tier],
        board=list(board or []),
        shop=[None, None, None],
        hand=[None for _ in range(HAND_SIZE)],
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
    )


def test_buff_random_friendly_no_others_is_noop():
    g = MiniBGGame(seed=0)
    buffer_card = make_minion("buffer")
    p = _player(board=[buffer_card])
    g._fire_on_place(buffer_card, p, None)
    assert buffer_card.bonus_attack == 0
    assert buffer_card.bonus_health == 0


def test_buff_random_friendly_picks_only_other_minion():
    g = MiniBGGame(seed=0)
    target = make_minion("murloc_warleader")
    buffer_card = make_minion("buffer")
    p = _player(board=[target, buffer_card])
    g._fire_on_place(buffer_card, p, None)
    assert target.bonus_attack == 1
    assert target.bonus_health == 1
    assert buffer_card.bonus_attack == 0
    assert buffer_card.bonus_health == 0


def test_mentor_on_turn_end_buffs_other_friendly():
    g = MiniBGGame(seed=0)
    mentor = make_minion("mentor")
    mech = make_minion("toy_mech")
    p = _player(board=[mentor, mech])
    g._fire_on_turn_end(p)
    assert mech.bonus_attack == 2
    assert mech.bonus_health == 2
    assert mech.card_id == "BOT_445"
    assert mentor.bonus_attack == 0
    assert mentor.bonus_health == 0


def test_mentor_on_turn_end_no_others_is_noop():
    g = MiniBGGame(seed=0)
    mentor = make_minion("mentor")
    p = _player(board=[mentor])
    g._fire_on_turn_end(p)
    assert mentor.bonus_attack == 0
    assert mentor.bonus_health == 0


def test_two_mentors_both_fire():
    g = MiniBGGame(seed=0)
    m1 = make_minion("mentor")
    m2 = make_minion("mentor")
    mech = make_minion("toy_mech")
    p = _player(board=[m1, m2, mech])
    g._fire_on_turn_end(p)
    total_atk = m1.bonus_attack + m2.bonus_attack + mech.bonus_attack
    total_hp = m1.bonus_health + m2.bonus_health + mech.bonus_health
    assert total_atk == 4
    assert total_hp == 4


def test_summon_effect_on_death_appends_token():
    pack_rat = make_minion("pack_rat")
    bm = BattleMinion.from_minion(pack_rat, 1)
    bm.current_health = 0
    side = BattleSide(minions=[bm])
    rt = _CombatRuntime(
        sides=(side, BattleSide()),
        rng=np.random.default_rng(0),
        combat_board_max=COMBAT_BOARD_MAX,
        damage_cap=DAMAGE_CAP,
    )
    _fire_deathrattle(rt, bm, 0)
    assert len(side.minions) == 3
    rats = [m for m in side.minions if m.template.card_id == "CFM_316t"]
    assert len(rats) == 2
    assert all(m.alive for m in rats)


def test_summon_effect_skipped_when_alive_count_at_cap():
    pack_rat = make_minion("pack_rat")
    bm = BattleMinion.from_minion(pack_rat, 1)
    bm.current_health = 0
    extras = [
        BattleMinion.from_minion(make_minion("recruit"), i)
        for i in range(2, 9)
    ]
    side = BattleSide(minions=[bm, *extras])
    rt = _CombatRuntime(
        sides=(side, BattleSide()),
        rng=np.random.default_rng(0),
        combat_board_max=COMBAT_BOARD_MAX,
        damage_cap=DAMAGE_CAP,
    )
    _fire_deathrattle(rt, bm, 0)
    rat_summons = [m for m in side.minions if m.template.card_id == "CFM_316t"]
    assert rat_summons == []


def test_the_beast_summons_finkle_on_opponent_side_during_dr():
    beast = BattleMinion.from_minion(make_minion("the_beast"), 1)
    beast.current_health = 0
    side0 = BattleSide(minions=[beast])
    side1 = BattleSide(minions=[])
    rt = _CombatRuntime(
        sides=(side0, side1),
        rng=np.random.default_rng(0),
        combat_board_max=COMBAT_BOARD_MAX,
        damage_cap=DAMAGE_CAP,
    )
    _fire_deathrattle(rt, beast, 0)
    assert len(side1.minions) == 1
    assert side1.minions[0].template.card_id == "EX1_finkle"


def test_opponent_summon_skipped_when_target_side_at_combat_cap():
    beast = BattleMinion.from_minion(make_minion("the_beast"), 1)
    beast.current_health = 0
    extras = [BattleMinion.from_minion(make_minion("recruit"), i) for i in range(2, 9)]
    side0 = BattleSide(minions=[beast])
    side1 = BattleSide(minions=extras)
    rt = _CombatRuntime(
        sides=(side0, side1),
        rng=np.random.default_rng(0),
        combat_board_max=COMBAT_BOARD_MAX,
        damage_cap=DAMAGE_CAP,
    )
    _fire_deathrattle(rt, beast, 0)
    assert not any(m.template.card_id == "EX1_finkle" for m in side1.minions)
    assert side1.alive_count() == 7


def test_malganis_tribal_aura_buffs_other_demons_only():
    mal = make_minion("mal_ganis")
    imp = make_minion("imp_demon")
    side = BattleSide(minions=[
        BattleMinion.from_minion(mal, 1),
        BattleMinion.from_minion(imp, 2),
    ])
    mal_b, imp_b = side.minions
    assert attack_with_auras(imp_b, side) == imp.raw_attack + 2
    assert attack_with_auras(mal_b, side) == mal.raw_attack


def test_two_malganis_buff_each_other_and_stack_on_third_demon():
    m1 = make_minion("mal_ganis")
    m2 = make_minion("mal_ganis")
    imp = make_minion("imp_demon")
    side = BattleSide(minions=[
        BattleMinion.from_minion(m1, 1),
        BattleMinion.from_minion(m2, 2),
        BattleMinion.from_minion(imp, 3),
    ])
    a, b, im = side.minions
    assert attack_with_auras(a, side) == m1.raw_attack + 2
    assert attack_with_auras(b, side) == m2.raw_attack + 2
    assert attack_with_auras(im, side) == imp.raw_attack + 4


def test_defender_argus_buffs_adjacent_in_shop():
    from src.bg_recruitment.targeted_battlecry import apply_targeted_on_place_battlecries

    g = MiniBGGame(seed=0)
    left = make_minion("recruit")
    right = make_minion("recruit")
    argus = make_minion("defender_argus")
    p = _player(board=[left, argus, right])
    apply_targeted_on_place_battlecries(g._shop_triggers, p, argus, rng=g._rng)
    assert left.bonus_attack == 1 and left.bonus_health == 1
    assert Keyword.TAUNT in left.keywords
    assert right.bonus_attack == 1 and Keyword.TAUNT in right.keywords
    assert argus.bonus_attack == 0 and argus.bonus_health == 0


def test_murloc_warleader_tribal_aura_in_combat():
    side = BattleSide(minions=[
        BattleMinion.from_minion(make_minion("murloc_warleader"), 1),
        BattleMinion.from_minion(make_minion("old_murk_eye"), 2),
        BattleMinion.from_minion(make_minion("toy_mech"), 3),
    ])
    wl, mur, filler = side.minions
    assert attack_with_auras(mur, side) == mur.raw_attack + 2 + 1
    assert attack_with_auras(wl, side) == wl.raw_attack
    assert attack_with_auras(filler, side) == filler.raw_attack


def test_phalanx_commander_taunt_keyword_aura():
    side = BattleSide(minions=[
        BattleMinion.from_minion(make_minion("phalanx_commander"), 1),
        BattleMinion.from_minion(make_minion("guard"), 2),
        BattleMinion.from_minion(make_minion("toy_mech"), 3),
    ])
    _, g, rec = side.minions
    assert attack_with_auras(g, side) == g.raw_attack + 2
    assert attack_with_auras(rec, side) == rec.raw_attack


def test_tribal_aura_drops_when_source_dies():
    mal = make_minion("mal_ganis")
    imp = make_minion("imp_demon")
    side = BattleSide(minions=[
        BattleMinion.from_minion(mal, 1),
        BattleMinion.from_minion(imp, 2),
    ])
    mal_b, imp_b = side.minions
    mal_b.current_health = 0
    assert attack_with_auras(imp_b, side) == imp.raw_attack


def test_kangor_deathrattle_uses_dead_mech_corpses_left_to_right():
    m1 = BattleMinion.from_minion(make_minion("toy_mech"), 1)
    m1.current_health = 0
    m2 = BattleMinion.from_minion(make_minion("shield_bot"), 2)
    m2.current_health = 0
    kang_tpl = make_minion("kangors_apprentice")
    kang_tpl.abilities = (
        Ability(Trigger.ON_DEATH, SummonFirstDeadFriendlyMechsThisCombat(count=2)),
    )
    kang = BattleMinion.from_minion(kang_tpl, 3)
    kang.current_health = 0
    side = BattleSide(minions=[m1, m2, kang])
    rt = _CombatRuntime(
        sides=(side, BattleSide()),
        rng=np.random.default_rng(0),
        combat_board_max=COMBAT_BOARD_MAX,
        damage_cap=DAMAGE_CAP,
    )
    _fire_deathrattle(rt, kang, 0)
    alive_ids = [m.template.card_id for m in side.minions if m.alive]
    assert alive_ids.count("BOT_445") == 1
    assert alive_ids.count("GVG_058") == 1


def test_golden_selfless_hero_grants_two_divine_shields():
    sh = make_minion("TB_BaconUps_014")
    a = BattleMinion.from_minion(make_minion("recruit"), 1)
    b = BattleMinion.from_minion(make_minion("recruit"), 2)
    dead = BattleMinion.from_minion(sh, 3)
    dead.current_health = 0
    side = BattleSide(minions=[a, b, dead])
    rt = _CombatRuntime(
        sides=(side, BattleSide()),
        rng=np.random.default_rng(1),
        combat_board_max=COMBAT_BOARD_MAX,
        damage_cap=DAMAGE_CAP,
    )
    _fire_deathrattle(rt, dead, 0)
    assert Keyword.SHIELD in a.template.keywords and a.shield_armed
    assert Keyword.SHIELD in b.template.keywords and b.shield_armed
