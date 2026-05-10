import pytest

from src.envs.minibg.cards import CARD_TEMPLATES, make_minion, shop_pool_for_tier


def test_all_expected_cards_present():
    expected = {
        "recruit", "guard", "buffer",
        "bruiser", "shield_bot", "pack_rat",
        "big_guy", "commander", "summoner",
        "rat_token", "summoned_token",
    }
    assert expected.issubset(CARD_TEMPLATES.keys())


def test_shield_bot_has_shield_armed():
    m = make_minion("shield_bot")
    assert m.has_shield is True


def test_non_shield_minion_has_no_shield():
    m = make_minion("recruit")
    assert m.has_shield is False


def test_tokens_marked_and_excluded_from_shop_pool():
    rat = make_minion("rat_token")
    summoned = make_minion("summoned_token")
    assert rat.is_token and summoned.is_token
    assert "rat_token" not in shop_pool_for_tier(3)
    assert "summoned_token" not in shop_pool_for_tier(3)


def test_shop_pool_per_tier():
    t1 = set(shop_pool_for_tier(1))
    t2 = set(shop_pool_for_tier(2))
    t3 = set(shop_pool_for_tier(3))
    assert t1 == {"recruit", "guard", "buffer"}
    assert t2 == t1 | {"bruiser", "shield_bot", "pack_rat"}
    assert t3 == t2 | {"big_guy", "commander", "summoner"}


def test_make_minion_returns_distinct_instances():
    a = make_minion("recruit")
    b = make_minion("recruit")
    assert a is not b
    a.bonus_attack += 5
    assert b.bonus_attack == 0


def test_make_minion_unknown_card_raises():
    with pytest.raises(KeyError):
        make_minion("not_a_real_card")
