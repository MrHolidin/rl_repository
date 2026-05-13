import pytest

from src.envs.minibg.cards import (
    CARD_TEMPLATES,
    LEGACY_CARD_ID_ALIASES,
    make_minion,
    resolve_card_id,
    shop_pool_for_tier,
)
from src.envs.minibg.replay import minion_to_dict


def test_legacy_aliases_map_into_card_templates():
    for legacy, canon in LEGACY_CARD_ID_ALIASES.items():
        assert canon in CARD_TEMPLATES
        assert resolve_card_id(legacy) == canon


def test_patch_tavern_pool_is_eighty_one_non_golden_minions():
    pool = [m for m in CARD_TEMPLATES.values() if not m.is_token and not m.is_golden]
    assert len(pool) == 81


def test_minion_has_catalog_display_name():
    m = make_minion("recruit")
    assert m.name == "Dire Wolf Alpha"
    assert minion_to_dict(m)["name"] == "Dire Wolf Alpha"


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
    ids = shop_pool_for_tier(6)
    assert "CFM_316t" not in ids
    assert "skele21" not in ids


def test_shop_pool_is_monotone_by_tier_and_excludes_goldens():
    ids6 = shop_pool_for_tier(6)
    assert all(not CARD_TEMPLATES[cid].is_golden for cid in ids6)
    assert all(not CARD_TEMPLATES[cid].is_token for cid in ids6)
    t1 = set(shop_pool_for_tier(1))
    t6 = set(ids6)
    assert t1.issubset(t6)
    for cid in t1:
        assert CARD_TEMPLATES[cid].tier == 1


def test_make_minion_returns_distinct_instances():
    a = make_minion("recruit")
    b = make_minion("recruit")
    assert a is not b
    a.bonus_attack += 5
    assert b.bonus_attack == 0


def test_golden_templates_are_separate_cards_with_explicit_flag():
    assert make_minion("brann").is_golden is False
    assert make_minion("brann_golden").is_golden is True
    assert make_minion("brann").card_id != make_minion("brann_golden").card_id


def test_make_minion_unknown_card_raises():
    with pytest.raises(KeyError):
        make_minion("not_a_real_card")
