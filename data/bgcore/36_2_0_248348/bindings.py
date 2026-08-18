"""Card ability bindings for patch 36.2.0 (build 248348).

Filled in tier order — a tier-1 card is reachable in every game, a tier-7 card
in few — leaning on the engine mechanics already in place: Blood Gems, Rally,
Spellcraft with its "until next turn" buffs, Activate, Choose One, Venomous,
Avenge, Lockbox, Fishbait.

``scripts/check_patch_coverage.py data/bgcore/36_2_0_248348`` lists every pool
card whose text promises something no binding delivers. The list is the queue;
it shrinks as bindings land, and the checker fails the day a binding names a
card the catalog does not have.

Tier 1 is done except for seven cards that need engine mechanics that do not
exist yet; each is named in ``UNBOUND_NEEDS_ENGINE`` below with what it wants.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Tuple

from src.bg_core.effects import (
    Ability,
    AddRandomMinionToHandEffect,
    BloodGemTarget,
    BuffAllShopOffersEffect,
    BuffSelf,
    BuffTargetFriendlyBattlecry,
    ChooseOneEffect,
    CreateSpellcraftSpellEffect,
    DealHeroDamage,
    DiscoverMinionAtTierEffect,
    GainBloodGemsEffect,
    GainGoldNextTurnEffect,
    GainGoldThisTurnEffect,
    GrantKeywordAtAttackThreshold,
    GrantTemporaryBuffEffect,
    Keyword,
    PlayBloodGemsEffect,
    ReduceTavernSpellCostEffect,
    StealTavernMinionEffect,
    SummonEffect,
    SummonSelfCopyFromHandEffect,
    Trigger,
)
from src.bg_core.minion import Race

#: Golden rewards ("Get a Discover of a higher tier") — none bound yet.
GOLDEN_REWARD_IDS: FrozenSet[str] = frozenset()

#: Tokens summoned by bound cards. Grows with the deathrattles that summon them.
TOKEN_IDS: FrozenSet[str] = frozenset(
    {
        "BG28_603t",  # Beetle 2/2 — Buzzing Vermin
        "BG_BOT_312t",  # Microbot 1/1 — Cord Puller
        "BG_ICC_026t",  # Skeleton 1/1 — Harmless Bonehead
        "BG36_200t",  # Foraging Bat 1/1 — Flittering Bat
    }
)

#: Cards that come out of the tavern already golden. The flag is the whole
#: rule: the copy renders golden, and the triple resolver only merges non-golden
#: copies, so three of them never combine into a Triple Reward.
ALWAYS_GOLDEN_POOL_IDS: FrozenSet[str] = frozenset(
    {
        "BG32_236",  # Aureate Laureate
    }
)

#: Pool cards whose whole text is keywords the catalog already carries
#: (a plain Taunt/Divine Shield body), so they need no binding to be correct.
KEYWORD_ONLY_POOL_IDS: FrozenSet[str] = frozenset(
    {
        "BGS_119",  # Crackling Cyclone — Divine Shield, Windfury
        "BG25_001",  # Risen Rider — Taunt, Reborn
    }
)

#: Pool cards left unbound because the mechanic they need does not exist in the
#: engine yet. Kept here rather than in a commit message so the next pass over a
#: tier can read what is missing without re-deriving it from card text.
#:
#: Empty through tier 1. (Duos-only cards are not listed: they never reach a
#: solo pool — see ``is_duos_only_card_id``.)
UNBOUND_NEEDS_ENGINE: Dict[str, str] = {}

EFFECTS: Dict[str, Tuple[Ability, ...]] = {
    # ------------------------------------------------------------------ tier 1
    "BGS_004": (  # Wrath Weaver — after you play a Demon, 1 damage to hero, +2/+2
        Ability(
            Trigger.AFTER_FRIENDLY_MINION_PLACED,
            DealHeroDamage(1),
            filter_race=Race.DEMON,
        ),
        Ability(
            Trigger.AFTER_FRIENDLY_MINION_PLACED,
            BuffSelf(attack=2, health=2),
            filter_race=Race.DEMON,
        ),
    ),
    "BGS_127": (  # Molten Rock — after you play an Elemental, gain +1 Health
        Ability(
            Trigger.AFTER_FRIENDLY_MINION_PLACED,
            BuffSelf(attack=0, health=1),
            filter_race=Race.ELEMENTAL,
        ),
    ),
    "BG31_803": (  # Buzzing Vermin — Deathrattle: summon a 2/2 Beetle
        Ability(Trigger.ON_DEATH, SummonEffect(token_id="BG28_603t", count=1)),
    ),
    "BG29_611": (  # Cord Puller — Deathrattle: summon a 1/1 Microbot
        Ability(Trigger.ON_DEATH, SummonEffect(token_id="BG_BOT_312t", count=1)),
    ),
    "BG28_300": (  # Harmless Bonehead — Deathrattle: summon two 1/1 Skeletons
        Ability(Trigger.ON_DEATH, SummonEffect(token_id="BG_ICC_026t", count=2)),
    ),
    "BG26_146": (  # Lullabot — Magnetic; at the end of your turn, gain +1 Health
        Ability(Trigger.ON_TURN_END, BuffSelf(attack=0, health=1)),
    ),
    "BG20_100": (  # Razorfen Geomancer — Battlecry: get 2 Blood Gems
        Ability(Trigger.ON_PLACE, GainBloodGemsEffect(count=2)),
    ),
    "BG23_000": (  # Mini-Myrmidon — Spellcraft: give a minion +2 Attack until next turn
        Ability(
            Trigger.ON_TURN_START,
            CreateSpellcraftSpellEffect(
                buff=GrantTemporaryBuffEffect(attack=2),
                card_id="BG23_000t",
                name="Myrmidon's Might",
            ),
        ),
    ),
    "BG36_200": (  # Flittering Bat — Rally: summon a 1/1 Beast
        Ability(Trigger.ON_ATTACK, SummonEffect(token_id="BG36_200t", count=1)),
    ),
    "BG29_888": (  # Glim Guardian — Rally: gain +2 Attack
        Ability(Trigger.ON_ATTACK, BuffSelf(attack=2, health=0)),
    ),
    "BG33_886": (  # Tusked Camper — Rally: this plays a Blood Gem on itself
        Ability(
            Trigger.ON_ATTACK,
            PlayBloodGemsEffect(target=BloodGemTarget.SELF, count=1),
        ),
    ),
    "BG25_013": (  # Rot Hide Gnoll — +1 Attack per friendly dead this combat
        # Written as a listener rather than an aura: deaths only ever accumulate
        # within a combat, so counting them as they happen gives the same number
        # the card's "for each" would, and combat copies are thrown away after,
        # which is what keeps it from carrying into the next fight.
        Ability(Trigger.ON_FRIENDLY_MINION_DIED, BuffSelf(attack=1, health=0)),
    ),
    "BG33_140": (  # River Skipper — when you sell this, get a random Tier 1 minion
        Ability(Trigger.ON_SELL, AddRandomMinionToHandEffect(tier=1)),
    ),
    "BG26_135": (  # Southsea Busker — Battlecry: gain 1 Gold next turn
        Ability(Trigger.ON_PLACE, GainGoldNextTurnEffect(amount=1)),
    ),
    "BG36_345": (  # Suspicious Prisonguard — Activate (1): give another minion +3/+3
        Ability(
            Trigger.ON_ACTIVATE,
            BuffTargetFriendlyBattlecry(attack=3, health=3, exclude_self=True),
            activate_cost=1,
        ),
    ),
    "BG36_921": (  # Fleeing Fugitive — whenever you cast a spell on this, +1 Health
        Ability(Trigger.ON_TARGETED_BY_SPELL, BuffSelf(attack=0, health=1)),
    ),
    "BG35_814": (  # Scarlet Survivor — once this reaches 6 Attack, gain Divine Shield
        Ability(
            Trigger.AURA,
            GrantKeywordAtAttackThreshold(threshold=6, keyword=Keyword.SHIELD),
        ),
    ),
    "BG32_330": (  # Flighty Scout — Start of Combat: if in hand, summon a copy
        Ability(Trigger.ON_START_OF_COMBAT, SummonSelfCopyFromHandEffect()),
    ),
    "BG31_330": (  # Ominous Seer — Battlecry: next Tavern spell costs (1) less
        Ability(Trigger.ON_PLACE, ReduceTavernSpellCostEffect(amount=1)),
    ),
}


#: Tavern spells, bound the same way minions are. A spell's whole text is its
#: battlecry, so every ability here hangs off ``Trigger.ON_PLACE`` — it fires
#: when the card is cast, which for a spell is the only thing it ever does.
SPELL_EFFECTS: Dict[str, Tuple[Ability, ...]] = {
    # ------------------------------------------------------------------ tier 1
    "BG28_810": (  # Tavern Coin — Gain 1 Gold
        Ability(Trigger.ON_PLACE, GainGoldThisTurnEffect(amount=1)),
    ),
    "BG28_503": (  # Fortify — give a minion +3 Health and Taunt
        Ability(
            Trigger.ON_PLACE,
            BuffTargetFriendlyBattlecry(
                attack=0, health=3, exclude_self=False, grant_keyword=Keyword.TAUNT
            ),
        ),
    ),
    "BG28_897": (  # Tavern Dish Banana — give a minion +2/+2
        Ability(
            Trigger.ON_PLACE,
            BuffTargetFriendlyBattlecry(attack=2, health=2, exclude_self=False),
        ),
    ),
    "BG28_966": (  # Them Apples — give minions in the Tavern +1/+2
        Ability(Trigger.ON_PLACE, BuffAllShopOffersEffect(attack=1, health=2)),
    ),
    "BG28_504": (  # Recruit a Trainee — get a random Tier 1 minion
        Ability(Trigger.ON_PLACE, AddRandomMinionToHandEffect(tier=1)),
    ),
    "BG28_512": (  # Enchanted Lasso — steal a random minion from the Tavern
        Ability(Trigger.ON_PLACE, StealTavernMinionEffect()),
    ),
    "BG33_101": (  # A New Sprout — Discover a Tier 1 minion
        Ability(Trigger.ON_PLACE, DiscoverMinionAtTierEffect(tier=1)),
    ),
    "BG31_880": (  # Alliance Flag — Choose One: give a minion +3/+1; or +1/+3
        Ability(
            Trigger.ON_PLACE,
            ChooseOneEffect(
                first=BuffTargetFriendlyBattlecry(attack=3, health=1, exclude_self=False),
                second=BuffTargetFriendlyBattlecry(attack=1, health=3, exclude_self=False),
            ),
        ),
    ),
}
