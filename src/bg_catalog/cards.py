from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Tuple

from src.bg_catalog.patch_context import PatchContext, require_patch
from src.bg_core.effects import Ability
from src.bg_core.minion import Minion, Race

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
    "target_buffer": "target_buffer",
    "triple_reward_discover": "triple_reward_discover",
    "finkle_einhorn": "EX1_finkle",
    "the_beast": "EX1_577",
}


def templates(*, patch: PatchContext) -> Mapping[str, Minion]:
    return patch.templates


def build_card_templates(*, patch: PatchContext) -> Dict[str, Minion]:
    return dict(templates(patch=patch))


def triple_merge_golden_abilities(
    normal_card_id: str,
    *,
    patch: PatchContext,
) -> tuple[Ability, ...]:
    return patch.triple_merge_golden_abilities(normal_card_id)


def resolve_card_id(card_id: str) -> str:
    return LEGACY_CARD_ID_ALIASES.get(card_id, card_id)


def make_minion(card_id: str, *, patch: PatchContext) -> Minion:
    ctx = require_patch(patch, where="cards.make_minion")
    return ctx.make_minion(resolve_card_id(card_id))


def normalize_shop_excluded_races(
    shop_excluded_race: Race | Iterable[Race] | None,
) -> Tuple[Race, ...]:
    if shop_excluded_race is None:
        return ()
    if isinstance(shop_excluded_race, Race):
        return (shop_excluded_race,)
    out: list[Race] = []
    seen: set[Race] = set()
    for race in shop_excluded_race:
        if race in seen:
            continue
        out.append(race)
        seen.add(race)
    return tuple(out)


def shop_minion_allowed_with_exclusion(
    m: Minion,
    shop_excluded_race: Race | Iterable[Race] | None,
) -> bool:
    if m.race is None or m.race == Race.ALL:
        return True
    excluded = normalize_shop_excluded_races(shop_excluded_race)
    if not excluded:
        return True
    return m.race not in excluded


def shop_pool_for_tier(
    tier: int,
    *,
    shop_excluded_race: Race | Iterable[Race] | None = None,
    patch: PatchContext,
) -> List[str]:
    ctx = require_patch(patch, where="cards.shop_pool_for_tier")
    return [
        cid
        for cid, m in ctx.templates.items()
        if not m.is_token
        and not m.is_golden
        and not m.is_triple_reward_spell
        and m.tier == tier
        and shop_minion_allowed_with_exclusion(m, shop_excluded_race)
    ]
