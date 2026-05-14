from src.envs.minibg.patch_catalog import (
    catalog_path,
    load_patch_catalog,
    load_tavern_minions,
    minion_by_dbf_id,
    minion_by_id,
    minion_from_tavern_record,
    patch_build,
    patch_version,
    tier_by_dbf_id,
)


def test_catalog_file_exists():
    assert catalog_path().is_file()


def test_patch_pins_build_and_patch_name():
    assert patch_build() == 36393
    assert patch_version() == "15.6.2"


def test_tavern_minion_count_and_tiers():
    data = load_patch_catalog()
    assert data["tavernMinionCount"] == 186
    tiers = {m.tier for m in load_tavern_minions()}
    assert tiers == {1, 2, 3, 4, 5, 6}


def test_hsjson_merge_nathrezim_overseer():
    m59186 = minion_by_dbf_id()[59186]
    assert m59186.id == "BGS_001"
    assert m59186.name == "Nathrezim Overseer"
    assert m59186.tier == 2
    assert m59186.is_bacon_pool is True
    assert m59186.is_golden is False
    assert m59186.golden_dbf_id == 59487
    m59487 = minion_by_dbf_id()[59487]
    assert m59487.id == "TB_BaconUps_062"
    assert m59487.is_golden is True
    assert m59487.tier == 2
    assert minion_from_tavern_record(m59487).is_golden is True
    assert minion_from_tavern_record(m59186).is_golden is False


def test_tier_map_covers_all_catalog_dbfs():
    by_dbf = minion_by_dbf_id()
    tmap = tier_by_dbf_id()
    assert tmap.keys() == by_dbf.keys()


def test_emperor_cobra_constructed_pool_has_tier():
    """Non-BGS minions still carry TECH_LEVEL in CardDefs (tavern offerings)."""
    m = minion_by_id()["EX1_170"]
    assert m.name == "Emperor Cobra"
    assert m.tier == 1
    assert m.is_bacon_pool is False


def test_minion_from_catalog_poison_and_race():
    from src.envs.minibg.effects import Keyword
    from src.envs.minibg.obs import (
        KEYWORD_OFFSET,
        TIER_OFFSET,
        encode_minion,
    )
    from src.envs.minibg.state import Race

    cobra = minion_by_id()["EX1_170"]
    m = minion_from_tavern_record(cobra)
    assert Keyword.POISONOUS in m.keywords
    assert m.race == Race.BEAST
    v = encode_minion(m)
    assert v[0] == 1.0
    assert v[TIER_OFFSET] == 1.0
    assert v[KEYWORD_OFFSET + 3] == 1.0
