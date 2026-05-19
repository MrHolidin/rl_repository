from __future__ import annotations

from copy import copy
from typing import Dict, List, Optional

from src.bg_catalog.card_pool import CARD_TEMPLATES as _CARD_TEMPLATES
from src.bg_core.effects import Keyword
from src.bg_core.minion import Minion, Race

CARD_TEMPLATES: Dict[str, Minion] = dict(_CARD_TEMPLATES)

# Legacy toy / slug ids from tests and older scripts → patch catalog ``card_id``.
LEGACY_CARD_ID_ALIASES: Dict[str, str] = {
    "recruit": "EX1_162",
    "guard": "CS2_065",
    "buffer": "UNG_073",
    "bruiser": "EX1_507",
    "shield_bot": "GVG_058",
    "pack_rat": "CFM_316",
    "big_guy": "GVG_062",
    "commander": "EX1_185",
    "summoner": "EX1_556",
    "mentor": "GVG_027",
    "rat_token": "CFM_316t",
    "summoned_token": "skele21",
    "imp_demon": "BRM_006t",
    "mal_ganis": "GVG_021",
    "wrath_weaver": "BGS_004",
    "vulgar_homunculus": "LOOT_013",
    "annihilan": "BGS_010",
    "toy_mech": "BOT_445",
    "micro_machine": "BGS_027",
    "mech_base_dr": "EX1_556",
    "annoy_o_module": "BOT_911",
    "annoy_o_module_golden": "TB_BaconUps_099",
    "magnetic_dr_rat": "BOT_312",
    "kangors_apprentice": "BGS_012",
    "brann": "LOE_077",
    "brann_golden": "TB_BaconUps_045",
    "baron_rivendare": "FP1_031",
    "baron_golden": "TB_BaconUps_055",
    "khadgar": "DAL_575",
    "khadgar_golden": "TB_BaconUps_034",
    "old_murk_eye": "EX1_062",
    "zapp_slywick": "BGS_022",
    "cave_hydra": "LOOT_078",
    "windfury_recruit": "BGS_022",
    "murloc_warleader": "EX1_507",
    "rockpool_hunter": "UNG_073",
    "primalfin_lookout": "BGS_020",
    "gentle_megasaur": "BGS_031",
    "gentle_megasaur_golden": "TB_BaconUps_084",
    "siegebreaker": "EX1_185",
    "phalanx_commander": "ULD_179",
    "dire_wolf_alpha": "EX1_162",
    "defender_argus": "EX1_093",
    "finkle_einhorn": "EX1_finkle",
    "the_beast": "EX1_577",
}


def resolve_card_id(card_id: str) -> str:
    return LEGACY_CARD_ID_ALIASES.get(card_id, card_id)


def make_minion(card_id: str) -> Minion:
    template = CARD_TEMPLATES[resolve_card_id(card_id)]
    fresh = copy(template)
    fresh.has_shield = Keyword.SHIELD in template.all_keywords
    fresh.is_golden = template.is_golden
    fresh.from_triple_merge = False
    return fresh


def shop_minion_allowed_with_exclusion(
    m: Minion, shop_excluded_race: Optional[Race]
) -> bool:
    if shop_excluded_race is None:
        return True
    if m.race is None or m.race == Race.ALL:
        return True
    return m.race != shop_excluded_race


def shop_pool_for_tier(
    tavern_tier: int,
    *,
    shop_excluded_race: Optional[Race] = None,
) -> List[str]:
    return [
        cid
        for cid, m in CARD_TEMPLATES.items()
        if not m.is_token
        and not m.is_golden
        and m.tier <= tavern_tier
        and shop_minion_allowed_with_exclusion(m, shop_excluded_race)
    ]


__all__ = [
    "CARD_TEMPLATES",
    "LEGACY_CARD_ID_ALIASES",
    "make_minion",
    "resolve_card_id",
    "shop_minion_allowed_with_exclusion",
    "shop_pool_for_tier",
]
