import numpy as np

from src.envs.minibg.battle import (
    _decide_first_side,
    _pick_target,
    attack_with_auras,
    build_battle_side,
    simulate_battle,
)
from src.envs.minibg.cards import make_minion


def _board(*card_ids):
    return [make_minion(cid) for cid in card_ids]


def _rng(seed=0):
    return np.random.default_rng(seed)


def test_empty_vs_empty_is_draw():
    dmg = simulate_battle([], [], p0_has_initiative=True, rng=_rng())
    assert dmg == (0, 0)


def test_empty_vs_filled_loses():
    dmg_p0, dmg_p1 = simulate_battle(
        [], _board("recruit"), p0_has_initiative=True, rng=_rng()
    )
    assert dmg_p0 == 2
    assert dmg_p1 == 0


def test_winner_damage_scales_with_winner_tavern_tier():
    dmg_lo, _ = simulate_battle(
        [],
        _board("recruit"),
        p0_has_initiative=True,
        rng=_rng(),
        p1_tavern_tier=1,
    )
    dmg_hi, _ = simulate_battle(
        [],
        _board("recruit"),
        p0_has_initiative=True,
        rng=_rng(),
        p1_tavern_tier=4,
    )
    assert dmg_lo == 2  # 1 + minion tier 1
    assert dmg_hi == 5  # 4 + minion tier 1


def test_pick_target_respects_taunt():
    side = build_battle_side(_board("bruiser", "guard", "recruit"))
    rng = _rng(0)
    for _ in range(100):
        target = _pick_target(side, rng)
        assert target.template.card_id == "CS2_065"


def test_pick_target_uniform_when_no_taunt():
    side = build_battle_side(_board("bruiser", "recruit"))
    rng = _rng(0)
    seen = set()
    for _ in range(100):
        seen.add(_pick_target(side, rng).template.card_id)
    assert seen == {"EX1_507", "EX1_162"}


def test_decide_first_side_more_minions_wins():
    s_more = build_battle_side(_board("recruit", "recruit"))
    s_less = build_battle_side(_board("recruit"))
    assert _decide_first_side(s_more, s_less, p0_has_initiative=False) == 0
    assert _decide_first_side(s_less, s_more, p0_has_initiative=True) == 1


def test_decide_first_side_initiative_breaks_tie():
    s = build_battle_side(_board("recruit"))
    s2 = build_battle_side(_board("recruit"))
    assert _decide_first_side(s, s2, p0_has_initiative=True) == 0
    assert _decide_first_side(s, s2, p0_has_initiative=False) == 1


def test_shield_blocks_first_hit_and_breaks():
    p0 = _board("shield_bot")
    p1 = _board("recruit")
    dmg_p0, dmg_p1 = simulate_battle(p0, p1, p0_has_initiative=True, rng=_rng())
    assert dmg_p0 == 0
    assert dmg_p1 == 3


def test_pack_rat_deathrattle_summons_token():
    p0 = _board("pack_rat")
    p1 = _board("recruit")
    dmg_p0, dmg_p1 = simulate_battle(p0, p1, p0_has_initiative=True, rng=_rng())
    assert dmg_p0 == 0
    assert dmg_p1 == 3


def test_commander_aura_buffs_kill():
    p0 = _board("commander", "recruit")
    p1 = _board("guard")
    dmg_p0, dmg_p1 = simulate_battle(p0, p1, p0_has_initiative=True, rng=_rng())
    assert dmg_p0 == 0
    assert dmg_p1 == 6


def test_damage_below_cap_three_big_survivors():
    p0 = _board("big_guy", "big_guy", "big_guy")
    p1 = []
    dmg_p0, dmg_p1 = simulate_battle(p0, p1, p0_has_initiative=True, rng=_rng())
    assert dmg_p0 == 0
    assert dmg_p1 == 10


def test_simulate_battle_fill_persistent_board_kwargs():
    p0 = _board("recruit")
    p1: list = []
    out0: list = []
    out1: list = []
    simulate_battle(
        p0,
        p1,
        p0_has_initiative=True,
        rng=_rng(0),
        p0_board_out=out0,
        p1_board_out=out1,
    )
    assert [m.card_id for m in out0] == ["EX1_162"]
    assert out1 == []


def test_persist_shop_board_respects_max_slots():
    p0 = _board("recruit", "recruit", "recruit")
    p1: list = []
    out0: list = []
    simulate_battle(
        p0,
        p1,
        p0_has_initiative=True,
        rng=_rng(0),
        p0_board_out=out0,
        max_board_slots=2,
    )
    assert len(out0) == 2
    assert [m.card_id for m in out0] == ["EX1_162", "EX1_162"]


def test_battle_does_not_mutate_player_boards():
    p0 = _board("recruit", "recruit")
    p1 = _board("big_guy")
    snap_p0 = [(m.card_id, m.base_attack, m.base_health, m.bonus_attack, m.bonus_health, m.has_shield) for m in p0]
    snap_p1 = [(m.card_id, m.base_attack, m.base_health, m.bonus_attack, m.bonus_health, m.has_shield) for m in p1]
    simulate_battle(p0, p1, p0_has_initiative=True, rng=_rng())
    after_p0 = [(m.card_id, m.base_attack, m.base_health, m.bonus_attack, m.bonus_health, m.has_shield) for m in p0]
    after_p1 = [(m.card_id, m.base_attack, m.base_health, m.bonus_attack, m.bonus_health, m.has_shield) for m in p1]
    assert snap_p0 == after_p0
    assert snap_p1 == after_p1


def test_simultaneous_kill_results_in_draw():
    p0 = _board("recruit")
    p1 = _board("recruit")
    dmg_p0, dmg_p1 = simulate_battle(p0, p1, p0_has_initiative=True, rng=_rng())
    assert dmg_p0 == 0 and dmg_p1 == 0


def test_death_log_side0_before_side1_on_simultaneous_kill():
    death_log: list = []
    simulate_battle(
        _board("recruit"),
        _board("recruit"),
        p0_has_initiative=True,
        rng=_rng(),
        death_log=death_log,
    )
    assert death_log == [(0, "EX1_162"), (1, "EX1_162")]


def test_poisonous_kills_minion_that_survives_initial_hit():
    from copy import copy

    from src.envs.minibg.effects import Keyword

    tank = copy(make_minion("recruit"))
    tank.keywords = frozenset()
    tank.base_attack = 2
    tank.base_health = 2

    snake = copy(make_minion("recruit"))
    snake.keywords = frozenset({Keyword.POISONOUS})
    snake.base_attack = 1
    snake.base_health = 10

    dmg_p0, dmg_p1 = simulate_battle(
        [snake], [tank], p0_has_initiative=True, rng=_rng()
    )
    assert dmg_p0 == 0
    assert dmg_p1 == 2


def test_shield_resets_between_battles():
    p0 = _board("shield_bot")
    p1 = _board("recruit")
    simulate_battle(p0, p1, p0_has_initiative=True, rng=_rng())
    assert p0[0].has_shield is True
    dmg_p0, dmg_p1 = simulate_battle(p0, p1, p0_has_initiative=True, rng=_rng())
    assert dmg_p1 == 3


def test_kangor_deathrattle_summons_first_dead_friendly_mech_corpses():
    """Kangor is literal DR: no summons until Kangor dies; then first N dead Mech corpses (LTR)."""
    tm = make_minion("toy_mech")
    tm.abilities = ()
    p0 = [tm, make_minion("kangors_apprentice")]
    p1 = _board("recruit")
    mech_log: list = []
    survivors: list = []
    simulate_battle(
        p0,
        p1,
        p0_has_initiative=True,
        rng=_rng(0),
        mech_death_log=mech_log,
        p0_survivors_out=survivors,
    )
    assert len(mech_log) == 1
    assert mech_log[0][1].card_id == "BOT_445"
    assert survivors == ["BGS_012"]


def test_pick_target_zapp_prefers_minimum_attack_among_legal():
    zapp_side = build_battle_side(_board("zapp_slywick"))
    zapp = zapp_side.minions[0]
    def_side = build_battle_side(_board("bruiser", "toy_mech"))
    rng = _rng(0)
    for _ in range(80):
        assert _pick_target(def_side, rng, zapp).template.card_id == "BOT_445"


def test_pick_target_zapp_respects_taunt_then_lowest_attack():
    from copy import copy

    from src.envs.minibg.effects import Keyword

    zapp_side = build_battle_side(_board("zapp_slywick"))
    zapp = zapp_side.minions[0]
    taunt_murl = copy(make_minion("rockpool_hunter"))
    taunt_murl.keywords = frozenset(taunt_murl.keywords | {Keyword.TAUNT})
    def_side = build_battle_side([make_minion("guard"), taunt_murl])
    rng = _rng(0)
    for _ in range(80):
        assert _pick_target(def_side, rng, zapp).template.card_id == "CS2_065"


def test_windfury_zapp_kills_two_recruits_one_activation():
    dmg_p0, dmg_p1 = simulate_battle(
        _board("zapp_slywick"),
        _board("recruit", "recruit"),
        p0_has_initiative=True,
        rng=_rng(0),
    )
    assert dmg_p0 == 0
    assert dmg_p1 == 7


def test_zapp_windfury_second_swing_retargets_after_first_kill():
    dmg_p0, dmg_p1 = simulate_battle(
        _board("zapp_slywick"),
        _board("buffer", "bruiser"),
        p0_has_initiative=True,
        rng=_rng(0),
    )
    assert dmg_p0 == 0
    assert dmg_p1 == 7


def test_cave_hydra_cleave_hits_adjacent_around_taunt():
    dmg_p0, dmg_p1 = simulate_battle(
        _board("cave_hydra"),
        _board("recruit", "guard", "recruit"),
        p0_has_initiative=True,
        rng=_rng(0),
    )
    assert dmg_p0 == 2
    assert dmg_p1 == 0


def test_attack_during_death_resolution_ignores_stat_aura():
    from src.envs.minibg.battle import attack_value, build_battle_side

    mal = make_minion("mal_ganis")
    imp = make_minion("imp_demon")
    side = build_battle_side([mal, imp])
    mal_bm, imp_bm = side.minions
    assert attack_value(imp_bm, side, death_resolution=False) == 3
    assert attack_value(imp_bm, side, death_resolution=True) == 1
    mal_bm.current_health = 0
    assert attack_value(imp_bm, side, death_resolution=False) == 1


def test_dire_wolf_buffs_immediate_board_neighbors():
    side = build_battle_side(_board("dire_wolf_alpha", "toy_mech"))
    wolf, rec = side.minions
    assert attack_with_auras(rec, side) == rec.raw_attack + 1
    assert attack_with_auras(wolf, side) == wolf.raw_attack


def test_tombstone_keeps_slots_dire_wolf_no_nearest_living_fallthrough():
    """BG combat: corpses occupy fixed slots; adjacency is index ±1 (not nearest living)."""
    from src.envs.minibg.battle import BattleMinion, BattleSide, attack_with_auras

    wolf = BattleMinion.from_minion(make_minion("dire_wolf_alpha"), 1)
    rat = BattleMinion.from_minion(make_minion("pack_rat"), 2)
    rec = BattleMinion.from_minion(make_minion("toy_mech"), 3)
    side = BattleSide(minions=[wolf, rat, rec])
    assert attack_with_auras(rat, side) == rat.raw_attack + 1
    rat.current_health = 0
    assert attack_with_auras(rec, side) == rec.raw_attack


def test_malganis_health_bonus_drops_when_aura_source_dies():
    from src.envs.minibg.battle import BattleMinion, BattleSide, _sync_health_aura_side, health_aura_bonus

    mal = BattleMinion.from_minion(make_minion("mal_ganis"), 1)
    imp = BattleMinion.from_minion(make_minion("imp_demon"), 2)
    side = BattleSide(minions=[mal, imp])
    _sync_health_aura_side(side, False)
    assert imp.current_health == 3
    mal.current_health = 0
    _sync_health_aura_side(side, False)
    assert health_aura_bonus(imp, side, death_resolution=False) == 0
    assert imp.current_health == 1
