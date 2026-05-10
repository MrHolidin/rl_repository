from src.envs.minibg.battle import BattleMinion, BattleSide, attack_with_auras
from src.envs.minibg.cards import make_minion
from src.envs.minibg.state import PlayerState
from src.envs.minibg.game import MiniBGGame


def _player(board=None, gold=10, tier=1):
    return PlayerState(
        health=20,
        gold=gold,
        tavern_tier=tier,
        board=list(board or []),
        shop=[None, None, None],
        shopping_finished=False,
        shop_actions_used=0,
    )


def test_buff_random_friendly_no_others_is_noop():
    g = MiniBGGame(seed=0)
    buffer_card = make_minion("buffer")
    p = _player(board=[buffer_card])
    g._fire_on_buy(buffer_card, p)
    assert buffer_card.bonus_attack == 0
    assert buffer_card.bonus_health == 0


def test_buff_random_friendly_picks_only_other_minion():
    g = MiniBGGame(seed=0)
    target = make_minion("recruit")
    buffer_card = make_minion("buffer")
    p = _player(board=[target, buffer_card])
    g._fire_on_buy(buffer_card, p)
    assert target.bonus_attack == 1
    assert target.bonus_health == 1
    assert buffer_card.bonus_attack == 0
    assert buffer_card.bonus_health == 0


def test_mentor_on_turn_end_buffs_other_friendly():
    g = MiniBGGame(seed=0)
    mentor = make_minion("mentor")
    recruit = make_minion("recruit")
    p = _player(board=[mentor, recruit])
    g._fire_on_turn_end(p)
    assert recruit.bonus_attack == 2
    assert recruit.bonus_health == 1
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
    rec = make_minion("recruit")
    p = _player(board=[m1, m2, rec])
    g._fire_on_turn_end(p)
    total_atk = m1.bonus_attack + m2.bonus_attack + rec.bonus_attack
    total_hp = m1.bonus_health + m2.bonus_health + rec.bonus_health
    assert total_atk == 4
    assert total_hp == 2


def test_summon_effect_on_death_appends_token():
    pack_rat = make_minion("pack_rat")
    bm = BattleMinion.from_minion(pack_rat)
    bm.current_health = 0
    side = BattleSide(minions=[bm])
    from src.envs.minibg.battle import _resolve_deaths
    _resolve_deaths(side)
    assert len(side.minions) == 2
    summoned = side.minions[1]
    assert summoned.template.card_id == "rat_token"
    assert summoned.alive


def test_summon_effect_skipped_when_alive_count_at_cap():
    pack_rat = make_minion("pack_rat")
    bm = BattleMinion.from_minion(pack_rat)
    bm.current_health = 0
    extras = [BattleMinion.from_minion(make_minion("recruit")) for _ in range(4)]
    side = BattleSide(minions=[bm, *extras])
    from src.envs.minibg.battle import _resolve_deaths
    _resolve_deaths(side)
    rat_summons = [m for m in side.minions if m.template.card_id == "rat_token"]
    assert rat_summons == []


def test_stat_aura_grants_attack_to_others_only():
    cmd = make_minion("commander")
    rec = make_minion("recruit")
    side = BattleSide(minions=[
        BattleMinion.from_minion(cmd),
        BattleMinion.from_minion(rec),
    ])
    cmd_b, rec_b = side.minions
    assert attack_with_auras(rec_b, side) == rec.raw_attack + 1
    assert attack_with_auras(cmd_b, side) == cmd.raw_attack


def test_two_commanders_buff_each_other():
    cmd1 = make_minion("commander")
    cmd2 = make_minion("commander")
    rec = make_minion("recruit")
    side = BattleSide(minions=[
        BattleMinion.from_minion(cmd1),
        BattleMinion.from_minion(cmd2),
        BattleMinion.from_minion(rec),
    ])
    c1, c2, r = side.minions
    assert attack_with_auras(c1, side) == cmd1.raw_attack + 1
    assert attack_with_auras(c2, side) == cmd2.raw_attack + 1
    assert attack_with_auras(r, side) == rec.raw_attack + 2


def test_aura_disappears_when_source_dies():
    cmd = make_minion("commander")
    rec = make_minion("recruit")
    side = BattleSide(minions=[
        BattleMinion.from_minion(cmd),
        BattleMinion.from_minion(rec),
    ])
    cmd_b, rec_b = side.minions
    cmd_b.current_health = 0
    assert attack_with_auras(rec_b, side) == rec.raw_attack
