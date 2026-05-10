import numpy as np

from src.envs.minibg.battle import (
    _build_side,
    _decide_first_side,
    _pick_target,
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
    assert dmg_p0 == 1
    assert dmg_p1 == 0


def test_pick_target_respects_taunt():
    side = _build_side(_board("bruiser", "guard", "recruit"))
    rng = _rng(0)
    for _ in range(100):
        target = _pick_target(side, rng)
        assert target.template.card_id == "guard"


def test_pick_target_uniform_when_no_taunt():
    side = _build_side(_board("bruiser", "recruit"))
    rng = _rng(0)
    seen = set()
    for _ in range(100):
        seen.add(_pick_target(side, rng).template.card_id)
    assert seen == {"bruiser", "recruit"}


def test_decide_first_side_more_minions_wins():
    s_more = _build_side(_board("recruit", "recruit"))
    s_less = _build_side(_board("recruit"))
    assert _decide_first_side(s_more, s_less, p0_has_initiative=False) == 0
    assert _decide_first_side(s_less, s_more, p0_has_initiative=True) == 1


def test_decide_first_side_initiative_breaks_tie():
    s = _build_side(_board("recruit"))
    s2 = _build_side(_board("recruit"))
    assert _decide_first_side(s, s2, p0_has_initiative=True) == 0
    assert _decide_first_side(s, s2, p0_has_initiative=False) == 1


def test_shield_blocks_first_hit_and_breaks():
    p0 = _board("shield_bot")
    p1 = _board("recruit")
    dmg_p0, dmg_p1 = simulate_battle(p0, p1, p0_has_initiative=True, rng=_rng())
    assert dmg_p0 == 0
    assert dmg_p1 == 2


def test_pack_rat_deathrattle_summons_token():
    p0 = _board("pack_rat")
    p1 = _board("recruit")
    dmg_p0, dmg_p1 = simulate_battle(p0, p1, p0_has_initiative=True, rng=_rng())
    assert dmg_p0 == 0
    assert dmg_p1 == 1


def test_commander_aura_buffs_kill():
    p0 = _board("commander", "recruit")
    p1 = _board("guard")
    dmg_p0, dmg_p1 = simulate_battle(p0, p1, p0_has_initiative=True, rng=_rng())
    assert dmg_p0 == 0
    assert dmg_p1 == 4


def test_damage_cap_at_seven():
    p0 = _board("big_guy", "big_guy", "big_guy")
    p1 = []
    dmg_p0, dmg_p1 = simulate_battle(p0, p1, p0_has_initiative=True, rng=_rng())
    assert dmg_p0 == 0
    assert dmg_p1 == 7


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


def test_shield_resets_between_battles():
    p0 = _board("shield_bot")
    p1 = _board("recruit")
    simulate_battle(p0, p1, p0_has_initiative=True, rng=_rng())
    assert p0[0].has_shield is True
    dmg_p0, dmg_p1 = simulate_battle(p0, p1, p0_has_initiative=True, rng=_rng())
    assert dmg_p1 == 2
