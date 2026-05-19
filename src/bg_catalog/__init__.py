"""Card catalog: patch JSON, template pool, factories."""

from .card_pool import (
    CARD_TEMPLATES,
    EFFECTS,
    GOLDEN_REWARD_IDS,
    TOKEN_IDS,
    build_card_templates,
    triple_merge_golden_abilities,
)
from .cards import (
    LEGACY_CARD_ID_ALIASES,
    make_minion,
    resolve_card_id,
    shop_minion_allowed_with_exclusion,
    shop_pool_for_tier,
)
from .patch_catalog import (
    TavernMinionRecord,
    catalog_path,
    golden_upgrade_card_id,
    keywords_for_tavern_record,
    load_patch_catalog,
    load_tavern_minions,
    minion_by_dbf_id,
    minion_by_id,
    minion_from_tavern_record,
    normal_to_golden_card_id_map,
    patch_build,
    patch_version,
    race_from_hs_string,
    tier_by_dbf_id,
)
from .triple_effects import implicit_triple_golden_effect, resolve_triple_forged_abilities

__all__ = [
    "CARD_TEMPLATES",
    "EFFECTS",
    "GOLDEN_REWARD_IDS",
    "TOKEN_IDS",
    "LEGACY_CARD_ID_ALIASES",
    "TavernMinionRecord",
    "build_card_templates",
    "catalog_path",
    "golden_upgrade_card_id",
    "implicit_triple_golden_effect",
    "keywords_for_tavern_record",
    "load_patch_catalog",
    "load_tavern_minions",
    "make_minion",
    "minion_by_dbf_id",
    "minion_by_id",
    "minion_from_tavern_record",
    "normal_to_golden_card_id_map",
    "patch_build",
    "patch_version",
    "race_from_hs_string",
    "resolve_card_id",
    "resolve_triple_forged_abilities",
    "shop_minion_allowed_with_exclusion",
    "shop_pool_for_tier",
    "tier_by_dbf_id",
    "triple_merge_golden_abilities",
]
