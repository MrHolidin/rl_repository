"""Shop-phase trigger dispatch (ON_BUY, ON_PLACE, turn hooks, battlecries)."""

from __future__ import annotations

from dataclasses import replace as dataclass_replace

from typing import Dict, Any, Callable, List, Optional

import numpy as np

from src.bg_catalog.cards import make_minion
from src.bg_catalog.patch_context import PatchContext
from src.bg_core.conditions import ability_condition_met, condition_met
from src.bg_core.effects import (
    AdaptAllMurlocsEffect,
    BuffHandMinionsEffect,
    GiveOwnStatsToHandEffect,
    GrantCombinedChooseOneEffect,
    RaiseGoldCapEffect,
    SpellsCastResponseEffect,
    AddRandomGoldenMinionEffect,
    AddRandomMinionOfCommonTribeEffect,
    BuffTargetPerGoldSpentEffect,
    MakeFriendlyGoldenEffect,
    AdaptSelfRandomEffect,
    AddRandomMinionToShopEffect,
    AddFromLastOpponentBoardEffect,
    AddRandomMinionToHandEffect,
    CreateSpellcraftSpellEffect,
    GainBloodGemsEffect,
    GrantTemporaryBuffEffect,
    IncreaseBloodGemBonusEffect,
    PlayBloodGemsEffect,
    AddTokenToHandEffect,
    BuffSelfPerCount,
    TransformIntoShopMinionEffect,
    MultiplierKind,
    BuffAdjacentBattlecry,
    BuffMatching,
    BuffTarget,
    BuffAllShopOffersEffect,
    BuffOnePerListedTribeFriendly,
    BuffRandomFriendly,
    BuffRandomUniqueTribeFriendlies,
    BuffSelf,
    BuffSelfFromHeroDamageTaken,
    BuffTargetFriendlyBattlecry,
    BuffTargetFromPiratesBoughtBattlecry,
    ChooseOneEffect,
    ConsumeFriendlyBattlecry,
    ConsumeTavernMinionEffect,
    DestroyFriendlyEffect,
    BuffSelfWhenFriendlyBattlecryPlaced,
    BuffSelfWhenFriendlyDeathrattlePlaced,
    BuffLeftmostRepeatedEffect,
    BuffRandomFriendlyFromPlacedTierEffect,
    DealHeroDamage,
    AddSharedTribeMinionEffect,
    CopyTavernMinionEffect,
    DealHeroDamagePerTierEffect,
    TransferAttackToRandomFriendlyEffect,
    DealDamageAllMinions,
    DealDamageLeftmostEnemyMinion,
    DealDamageRandomEnemyMinion,
    BuffRandomOtherFriendlyCombat,
    SummonFirstDeadFriendlyMechsThisCombat,
    SummonRandomAndCopyToHandEffect,
    SummonRandomMinionEffect,
    DiscoverTribeEffect,
    Effect,
    GainGoldThisTurnEffect,
    GainGoldNextTurnEffect,
    BuffPlacedMinionEffect,
    AddRandomCardToHandEffect,
    BuffBoughtMinionEffect,
    BuffPerMagnetizationEffect,
    GoldSpentResponseEffect,
    RetriggerFriendlyAbilityEffect,
    ElementalsPlayedResponseEffect,
    CopyTargetingSpellEffect,
    GainStatsFromTavernEffect,
    SetArmorEffect,
    SetStatsEffect,
    StatsFromNextBuyEffect,
    MagnetizeTokenEffect,
    BuffSelfOnFriendlySoldEffect,
    DoubleNextMagnetizeEffect,
    BuffShopOnEveryRefreshEffect,
    BumpSeatCounterEffect,
    AddCardToNextRefreshesEffect,
    AddRandomTavernSpellToHandEffect,
    CastRandomTavernSpellEffect,
    CopyLastTavernSpellEffect,
    SummonBestFromHandEffect,
    SummonStashedEffect,
    DestroyKillerEffect,
    BuffRandomHandMinionEffect,
    CastSpellAtEffect,
    DiscoverMinionAtTierEffect,
    DiscoverTavernSpellEffect,
    IncreaseTavernSpellBonusEffect,
    RaiseStandingBonusEffect,
    RepeatPerCountEffect,
    StealTavernMinionEffect,
    GiveLockboxEffect,
    AddTavernSpellToHandEffect,
    ReduceTavernSpellCostEffect,
    GrantKeywordRandomFriendly,
    HeroImmuneAura,
    IncreaseTribeGiftEffect,
    IncrementShopTribeBonusEffect,
    Keyword,
    PlaceFishbaitEffect,
    PogoHopperBattlecry,
    ReduceUpgradeCostEffect,
    SetNextRollCostEffect,
    SummonEffect,
    Trigger,
)
from src.bg_recruitment.hand_slots import first_free_hand_slot
from src.bg_recruitment.shop_auras import refresh_attack_thresholds
from src.bg_recruitment.choose_one import grant_combined_choose_one, open_choose_one
from src.bg_recruitment.lockbox import give_lockbox, tick_lockboxes
from src.bg_recruitment.game_counts import (
    DEATHRATTLES_TRIGGERED,
    bump_seat_counter,
    bump_summoned,
    improve_level,
)
from src.bg_recruitment.fishbait import fire_tavern_rally
from src.bg_recruitment.standing_bonuses import (
    BonusScope,
    ScopeKind,
    raise_standing_bonus,
    settle_standing_bonuses,
)
from src.bg_recruitment.tavern_spells import steal_tavern_minion
from src.bg_recruitment.spellcraft import (
    discard_spellcraft_spells,
    expire_temporary_buffs,
    give_spellcraft_spell,
)
from src.bg_recruitment.activate import reset_activations
from src.bg_recruitment.economy import reset_health_refreshes
from src.bg_recruitment.blood_gems import (
    blood_gem_targets,
    give_blood_gems,
    play_blood_gem_on,
)
from src.bg_core.board_helpers import (
    count_for_source,
    minion_matches_tribe,
    apply_buff_matching,
    apply_buff_self,
    apply_summoned_listener,
    set_minion_stats,
    grant_keyword,
    grant_keyword_random,
    apply_buff_self_per_count,
    multiplier_for,
    count_unique_tribes,
)
from src.bg_core.minion import ALL_TRIBES, Minion, Race, is_locked
from src.envs.minibg.actions import BOARD_SIZE
from src.bg_recruitment.discover_pool import (
    ADAPT_KEYS_ALL,
    apply_adapt_key_to_minion,
    roll_adapt_triple,
    roll_discover_tribe_triple,
)
from src.bg_recruitment.shop import (
    add_random_minion_to_hand,
    add_random_minion_to_shop,
    buff_all_shop_offers,
    buff_shop_minions_of_tribe,
)
from src.bg_lobby.player import PendingChoice, PendingChoiceKind, PlayerState


class UnhandledShopEffect(RuntimeError):
    """An effect reached the shop dispatcher and nothing knew what to do with it.

    This used to be a silent fall-through, which made "does nothing in the
    tavern, by design" and "nobody ever implemented this" look identical from
    the outside -- including to the card that quietly stopped working.
    """


#: Effects that legitimately reach ``apply_shop_effect`` and must do nothing
#: there, because something upstream already applied them. Each entry is a
#: claim that can be checked, which is the point of writing them down.
#: Effects the Tavern-spell module owns end to end — including the tier
#: Discover, which is printed on a spell and on a minion alike.
_TAVERN_SPELL_EFFECTS = (
    CastSpellAtEffect,
    DiscoverMinionAtTierEffect,
    IncreaseTavernSpellBonusEffect,
    AddRandomTavernSpellToHandEffect,
    DiscoverTavernSpellEffect,
    CastRandomTavernSpellEffect,
    CopyLastTavernSpellEffect,
)

#: Deathrattle effects that only mean something inside a fight. A minion
#: destroyed in the tavern skips these rather than handing them to a dispatcher
#: with no board to summon onto and no enemy to damage.
_COMBAT_ONLY_ON_DEATH = (
    SummonEffect,
    SummonRandomMinionEffect,
    SummonRandomAndCopyToHandEffect,
    SummonFirstDeadFriendlyMechsThisCombat,
    BuffRandomOtherFriendlyCombat,
    DealDamageRandomEnemyMinion,
    DealDamageLeftmostEnemyMinion,
    DealDamageAllMinions,
    TransferAttackToRandomFriendlyEffect,
    # The rest of the same family, found by destroying a friendly in the tavern
    # (Butchering, Maw Caster, Disguised Graverobber) and watching the
    # dispatcher raise: a body summoned from hand or stashed has nowhere to go,
    # and there is no killer to punish outside a fight.
    SummonBestFromHandEffect,
    SummonStashedEffect,
    DestroyKillerEffect,
    BuffRandomHandMinionEffect,
)

_HANDLED_ELSEWHERE = (
    # Battlecries needing a target the player picked: applied by
    # bg_recruitment/targeted_battlecry.py off the placement action.
    BuffAdjacentBattlecry,
    BuffTargetFriendlyBattlecry,
    BuffTargetFromPiratesBoughtBattlecry,
    ChooseOneEffect,
    ConsumeFriendlyBattlecry,
    DestroyFriendlyEffect,
    BuffTargetPerGoldSpentEffect,
    MakeFriendlyGoldenEffect,
    # fire_on_sell applies this one itself: it belongs to a minion staying
    # behind rather than to the card being sold, and the dispatcher only ever
    # sees the card that leaves.
    BuffSelfOnFriendlySoldEffect,
    # fire_on_place applies these itself and then skips them here: they read
    # and write per-turn counters that Brann must not multiply.
    PogoHopperBattlecry,
    AdaptSelfRandomEffect,
    # fire_after_friendly_minion_placed handles these before it delegates.
    BuffPlacedMinionEffect,
    MagnetizeTokenEffect,
    BuffSelfPerCount,
    BuffSelfWhenFriendlyBattlecryPlaced,
    BuffSelfWhenFriendlyDeathrattlePlaced,
    BuffRandomFriendlyFromPlacedTierEffect,
    # Open a discover rather than resolve: fire_on_place sets pending_choice.
    AdaptAllMurlocsEffect,
)


def _most_common_tribe(player: PlayerState) -> Optional[Race]:
    """The tribe the seat has most of, ties broken by board order.

    An Amalgam counts for every tribe, which is what being one means — so a
    board of two Beasts and an Amalgam is three Beasts as far as this is
    concerned.
    """
    counts: Dict[Race, int] = {}
    order: List[Race] = []
    for minion in player.board:
        for tribe in ALL_TRIBES:
            if minion_matches_tribe(minion, tribe):
                if tribe not in counts:
                    order.append(tribe)
                counts[tribe] = counts.get(tribe, 0) + 1
    if not counts:
        return None
    best = max(counts.values())
    return next(t for t in order if counts[t] == best)


def _buff_hand_minions(player: PlayerState, effect) -> None:
    """Pay the minions in hand this effect reaches.

    Left-most, one tribe, or everything there — and optionally the board too,
    for the card that says "in your hand and board" in one breath.
    """
    held = [c for c in player.hand if isinstance(c, Minion) and not is_locked(c)]
    if effect.tribe is not None:
        held = [c for c in held if minion_matches_tribe(c, effect.tribe)]
    if effect.leftmost:
        held = held[:1]
    targets = list(held)
    if effect.also_board:
        targets += [
            m
            for m in player.board
            if effect.tribe is None or minion_matches_tribe(m, effect.tribe)
        ]
    for card in targets:
        card.bonus_attack += effect.attack
        card.bonus_health += effect.health


def _amplified_by_gift(player: PlayerState, source: Optional[Minion], effect):
    """Bigger, when the seat has been told this source's tribe gives extra.

    "Your Elementals give an extra +2 Health this game" is not a buff of its own
    and not a bonus on the tavern — it raises *whatever an Elemental gives*. Ten
    Elementals in this pool give stats through nine different effect types, so
    the amplifier reads the source's tribe and the effect's own numbers rather
    than naming any of them.

    Raising the gift is not itself a gift, or Sand Swirler would amplify its own
    grant.
    """
    gift_attack = player.elemental_gift_attack
    gift_health = player.elemental_gift_health
    if source is None or not (gift_attack or gift_health):
        return effect
    if isinstance(effect, IncreaseTribeGiftEffect):
        return effect
    if not minion_matches_tribe(source, Race.ELEMENTAL):
        return effect
    declared = getattr(effect, "__dataclass_fields__", {})
    if "attack" not in declared or "health" not in declared:
        return effect
    return dataclass_replace(
        effect,
        attack=effect.attack + gift_attack,
        health=effect.health + gift_health,
    )


class ShopTriggers:
    def __init__(
        self,
        rng: np.random.Generator,
        *,
        on_triples: Optional[Callable[[PlayerState], None]] = None,
        patch: PatchContext,
    ) -> None:
        from src.bg_catalog.patch_context import require_patch

        self._rng = rng
        self._on_triples = on_triples
        self._patch = require_patch(patch, where="ShopTriggers.__init__")

    def _resolve_triples(
        self, player: PlayerState, *, shared_pool: Any = None
    ) -> None:
        if self._on_triples is not None:
            self._on_triples(player)
            return
        from src.bg_recruitment import triples as recruitment_triples

        recruitment_triples.resolve_triples_loop(
            player, shared_pool=shared_pool, patch=self._patch
        )

    @staticmethod
    def minion_matches_tribe(m: Minion, tribe: Any) -> bool:
        if m.race is None:
            return False
        if tribe == Race.ALL or m.race == Race.ALL:
            return True
        return m.race == tribe

    @staticmethod
    def player_has_hero_immune(player: PlayerState) -> bool:
        for m in player.board:
            for ab in m.abilities:
                if ab.trigger == Trigger.AURA and isinstance(ab.effect, HeroImmuneAura):
                    return True
        return False

    def damage_hero(self, player: PlayerState, amount: int) -> None:
        if amount <= 0 or self.player_has_hero_immune(player):
            return
        from src.bg_lobby.player import apply_hero_damage

        # Through the one choke point, so a card that costs the seat life and a
        # lost combat land the same way: armor absorbs first and the listeners
        # that read hero damage read this too. Lich Baz'hial's power is the
        # reason he starts with fifteen armor.
        apply_hero_damage(player, amount, patch=self._patch)
        player.hero_damage_taken_total += amount

    def apply_buff_adjacent(
        self,
        player: PlayerState,
        source: Minion,
        effect: BuffAdjacentBattlecry,
    ) -> None:
        board = player.board
        try:
            idx = board.index(source)
        except ValueError:
            return
        for j in (idx - 1, idx + 1):
            if 0 <= j < len(board):
                tgt = board[j]
                tgt.bonus_attack += effect.attack
                tgt.bonus_health += effect.health
                if effect.grant_taunt:
                    tgt.keywords = frozenset(tgt.keywords | {Keyword.TAUNT})

    def apply_buff_random(
        self,
        source: Minion,
        effect: BuffRandomFriendly,
        board: List[Minion],
    ) -> None:
        for _ in range(max(1, effect.repeats)):
            pool = (
                [m for m in board if m is not source]
                if effect.exclude_self
                else list(board)
            )
            if effect.filter_race is not None:
                pool = [
                    m for m in pool if self.minion_matches_tribe(m, effect.filter_race)
                ]
            if not pool:
                return
            target = pool[int(self._rng.integers(0, len(pool)))]
            target.bonus_attack += effect.attack
            target.bonus_health += effect.health
            if effect.grant_taunt:
                target.keywords = frozenset(target.keywords | {Keyword.TAUNT})

    def apply_buff_one_per_listed_tribe(
        self,
        source: Minion,
        effect: BuffOnePerListedTribeFriendly,
        board: List[Minion],
    ) -> None:
        # "A friendly minion **of each type**" is one body per type, and each
        # body answers for one type only: an All-type minion was being picked
        # for all nine, which is +18/+18 off a two-gold spell.
        paid: List[Minion] = []
        for tribe in effect.tribes or ALL_TRIBES:
            pool = (
                [m for m in board if m is not source]
                if effect.exclude_self
                else list(board)
            )
            pool = [
                m
                for m in pool
                if self.minion_matches_tribe(m, tribe)
                and not any(m is already for already in paid)
            ]
            if not pool:
                continue
            target = pool[int(self._rng.integers(0, len(pool)))]
            target.bonus_attack += effect.attack
            target.bonus_health += effect.health
            paid.append(target)

    def apply_buff_random_unique_tribe(
        self,
        source: Minion,
        effect: BuffRandomUniqueTribeFriendlies,
        board: List[Minion],
    ) -> None:
        pool = (
            [m for m in board if m is not source]
            if effect.exclude_self
            else list(board)
        )
        by_tribe: dict[Race, List[Minion]] = {}
        for m in pool:
            if m.race is None or m.race == Race.ALL:
                continue
            by_tribe.setdefault(m.race, []).append(m)
        tribes = list(by_tribe.keys())
        if not tribes:
            return
        order = tribes.copy()
        for i in range(len(order) - 1, 0, -1):
            j = int(self._rng.integers(0, i + 1))
            order[i], order[j] = order[j], order[i]
        for tribe in order[: max(0, effect.count)]:
            candidates = by_tribe[tribe]
            target = candidates[int(self._rng.integers(0, len(candidates)))]
            target.bonus_attack += effect.attack
            target.bonus_health += effect.health

    def apply_buff_matching(
        self,
        player: PlayerState,
        source: Minion,
        effect: BuffMatching,
    ) -> None:
        apply_buff_matching(effect, player.board, source, rng=self._rng)

    def apply_grant_keyword_random(
        self,
        player: PlayerState,
        source: Minion,
        effect: GrantKeywordRandomFriendly,
    ) -> None:
        grant_keyword_random(
            effect,
            player.board,
            source,
            rng=self._rng,
            grant=grant_keyword,
        )

    def apply_summon_from_place(
        self, player: PlayerState, source: Minion, effect: SummonEffect
    ) -> None:
        if effect.for_opponent or effect.count_from_source_attack:
            return
        try:
            idx = player.board.index(source)
        except ValueError:
            return
        insert_at = idx + 1
        # Khadgar multiplies every summon, in the tavern as well as in combat
        # (combat does the same via auras._summon_multiplier).
        n_sum = self.summon_multiplier(player.board)
        for _ in range(max(0, effect.count) * n_sum):
            if len(player.board) >= BOARD_SIZE:
                break
            tok = make_minion(effect.token_id, patch=self._patch)
            player.board.insert(insert_at, tok)
            self.fire_shop_friendly_summoned(player, tok)
            insert_at += 1
        if player.pending_choice is None:
            self._resolve_triples(player)

    def fire_on_bought(self, player: PlayerState, bought: Minion) -> None:
        """The seat just bought ``bought`` — pay the watchers of a purchase.

        Separate from ON_FRIENDLY_BOUGHT, which counts Pirates: these cards act
        *on the card bought*, and one of them was promised it in advance.
        """
        for watcher in list(player.board):
            if watcher.wants_next_buy_stats:
                factor = max(1, int(watcher.wants_next_buy_stats))
                watcher.wants_next_buy_stats = 0
                watcher.bonus_attack += bought.raw_attack * factor
                watcher.bonus_health += bought.max_health * factor
            for ab in watcher.abilities:
                eff = ab.effect
                if not isinstance(eff, BuffBoughtMinionEffect):
                    continue
                if eff.once_per_turn and watcher.buy_answered_this_turn:
                    continue
                watcher.buy_answered_this_turn = True
                bought.bonus_attack += eff.attack
                bought.bonus_health += eff.health
                extra = max(1, int(eff.stat_multiplier)) - 1
                if extra:
                    bought.bonus_attack += bought.raw_attack * extra
                    bought.bonus_health += bought.max_health * extra

    def fire_spell_cast(self, player: PlayerState) -> None:
        """Count a spell against the watchers that answer every Nth.

        Distinct from the tally the "improved by every 4 spells" cards read:
        that one is the seat's and never resets, this one is each watcher's own
        countdown, which refills.
        """
        for watcher in list(player.board):
            for ab in watcher.abilities:
                eff = ab.effect
                if not isinstance(eff, SpellsCastResponseEffect):
                    continue
                watcher.spells_seen += 1
                times, watcher.spells_seen = divmod(
                    watcher.spells_seen, max(1, int(eff.threshold))
                )
                for _ in range(times):
                    if eff.effect is not None:
                        self.apply_shop_effect(player, watcher, eff.effect, placed=None)

    def fire_gold_spent(self, player: PlayerState, amount: int) -> None:
        """Count gold leaving the seat, and answer every full threshold.

        The running total is each watcher's own, because "(5 Gold left!)" is a
        countdown printed on the card and two copies count separately.
        """
        if amount <= 0:
            return
        player.gold_spent_this_turn += int(amount)
        for watcher in list(player.board):
            for ab in watcher.abilities:
                eff = ab.effect
                if not isinstance(eff, GoldSpentResponseEffect):
                    continue
                watcher.gold_spent_seen += int(amount)
                times, watcher.gold_spent_seen = divmod(
                    watcher.gold_spent_seen, max(1, int(eff.threshold))
                )
                for _ in range(times * max(1, int(eff.repeats))):
                    if eff.effect is not None:
                        self.apply_shop_effect(player, watcher, eff.effect, placed=None)

    def _fire_hand_watchers(self, player: PlayerState, placed: Minion) -> None:
        """Cards that watch from *hand* ("while this is in your hand, …").

        The only listeners in the game that are not on the board, so they are a
        pass of their own rather than a branch in the board loop — and the
        effect is applied to the card in hand, which is where it is standing.
        """
        for card in player.hand:
            if not isinstance(card, Minion) or is_locked(card):
                continue
            for ab in card.abilities:
                if ab.trigger is not Trigger.WHILE_IN_HAND:
                    continue
                # minion_matches_tribe, not ==: an Amalgam is every tribe, and
                # a bare comparison is how six of them stopped counting as any.
                if ab.filter_race is not None and not minion_matches_tribe(
                    placed, ab.filter_race
                ):
                    continue
                if ab.filter_max_tier and placed.tier > ab.filter_max_tier:
                    continue
                self.apply_shop_effect(player, card, ab.effect, placed)

    def fire_magnetized(
        self, player: PlayerState, target: Minion, magnet: Minion
    ) -> None:
        """A Magnetization just landed on ``target``.

        Shares AFTER_FRIENDLY_MINION_PLACED with playing a Mech, because the one
        card that watches both says so in one breath: "whenever you play or
        Magnetize a Mech, give it +3/+1". The effect is the same either way —
        the newcomer is paid — so the trigger is the same and the site is what
        differs.
        """
        for watcher in list(player.board):
            if watcher is target:
                continue
            for ab in watcher.abilities:
                if ab.trigger != Trigger.AFTER_FRIENDLY_MINION_PLACED:
                    continue
                if ab.filter_race is not None and target.race != ab.filter_race:
                    continue
                if isinstance(ab.effect, BuffPlacedMinionEffect):
                    target.bonus_attack += ab.effect.attack
                    target.bonus_health += ab.effect.health

    def fire_shop_friendly_summoned(self, player: PlayerState, summoned: Minion) -> None:
        # Every arrival is a summon as the cards use the word — played from
        # hand, summoned by a deathrattle, opened out of a Lockbox — and this
        # is the one place all of them come through.
        bump_summoned(player, summoned)
        for m in player.board:
            if m is summoned:
                continue
            for ab in m.abilities:
                if ab.trigger != Trigger.ON_FRIENDLY_MINION_SUMMONED:
                    continue
                # The one flag no other shop site honours, because BGS_071 is
                # the only ability carrying it and this is its trigger.
                if ab.combat_only:
                    continue
                apply_summoned_listener(
                    ab.effect, m, summoned, grant_keyword=grant_keyword
                )

    def apply_shop_effect(
        self,
        player: PlayerState,
        source: Minion,
        effect: Effect,
        placed: Optional[Minion],
        *,
        shop_excluded_race: Optional[Race] = None,
        shared_pool=None,
    ) -> None:
        if isinstance(effect, _HANDLED_ELSEWHERE):
            return
        effect = _amplified_by_gift(player, source, effect)
        if isinstance(effect, BuffRandomFriendly):
            self.apply_buff_random(source, effect, player.board)
        elif isinstance(effect, BuffOnePerListedTribeFriendly):
            self.apply_buff_one_per_listed_tribe(source, effect, player.board)
        elif isinstance(effect, BuffRandomUniqueTribeFriendlies):
            self.apply_buff_random_unique_tribe(source, effect, player.board)
        elif isinstance(effect, PlaceFishbaitEffect):
            self.refresh_with_fishbait(
                player,
                source,
                shop_excluded_race=shop_excluded_race,
                shared_pool=shared_pool,
            )
        elif isinstance(effect, BuffAllShopOffersEffect):
            buff_all_shop_offers(
                player,
                attack=effect.attack,
                health=effect.health,
                keyword_choices=effect.keyword_choices,
                rng=self._rng,
            )
        elif isinstance(effect, AddRandomMinionToShopEffect):
            add_random_minion_to_shop(
                player,
                effect.tribe,
                shop_excluded_race,
                rng=self._rng,
                shared_pool=shared_pool,
                patch=self._patch,
                freeze_slot=effect.freeze_slot,
            )
        elif isinstance(effect, DealHeroDamage):
            self.damage_hero(player, effect.amount)
        elif isinstance(effect, BuffSelf):
            apply_buff_self(source, effect)
        elif isinstance(effect, BuffSelfFromHeroDamageTaken):
            source.bonus_health += (
                player.hero_damage_taken_total * effect.health_per_damage
            )
        elif isinstance(effect, BuffMatching):
            self.apply_buff_matching(player, source, effect)
        elif isinstance(effect, GrantKeywordRandomFriendly):
            self.apply_grant_keyword_random(player, source, effect)
        elif isinstance(effect, SummonEffect):
            self.apply_summon_from_place(player, source, effect)
        elif isinstance(effect, ReduceUpgradeCostEffect):
            player.upgrade_cost_delta -= effect.amount
        elif isinstance(effect, SetNextRollCostEffect):
            player.next_roll_cost_override = effect.cost
            player.free_roll_charges = effect.uses
        elif isinstance(effect, GainGoldThisTurnEffect):
            if effect.filter_race is None or (
                placed is not None and self.minion_matches_tribe(placed, effect.filter_race)
            ):
                player.gold += effect.amount
        elif isinstance(effect, StealTavernMinionEffect):
            for _ in range(max(1, int(effect.count))):
                steal_tavern_minion(
                    player,
                    rng=self._rng,
                    shared_pool=shared_pool,
                    highest_attack=effect.highest_attack,
                    double_health=effect.double_health,
                    set_attack=effect.set_attack,
                    set_health=effect.set_health,
                )
        elif isinstance(effect, BumpSeatCounterEffect):
            bump_seat_counter(player, effect.counter)
        elif isinstance(effect, RepeatPerCountEffect):
            if effect.counter:
                # An "improves" card: worth its printed value once, plus once
                # more per `per` events counted.
                repeats = improve_level(player, effect.counter, effect.per)
            else:
                repeats = int(effect.base_repeats) + count_for_source(
                    effect.source, player.board, tribe=effect.tribe
                )
            inner = effect.effect
            for _ in range(max(0, repeats)):
                if isinstance(inner, BuffAdjacentBattlecry):
                    # Adjacency is positional, and the dispatcher hands
                    # BuffAdjacentBattlecry to the placement path (it is in
                    # _HANDLED_ELSEWHERE) because at placement the slot is not
                    # settled yet. Here the source is already standing, so the
                    # neighbours are simply known.
                    self.apply_buff_adjacent(player, source, inner)
                    continue
                self.apply_shop_effect(
                    player,
                    source,
                    inner,
                    placed,
                    shop_excluded_race=shop_excluded_race,
                    shared_pool=shared_pool,
                )
        elif isinstance(effect, _TAVERN_SPELL_EFFECTS):
            # Owned by bg_recruitment.tavern_spells: they read and write the
            # spell counter, the spell bonus and the seat's spell memory, none
            # of which this dispatcher knows about.
            from src.bg_recruitment.tavern_spells import apply_tavern_spell_effect

            apply_tavern_spell_effect(
                player,
                effect,
                rng=self._rng,
                patch=self._patch,
                source=source,
                shop_excluded_race=shop_excluded_race,
                shared_pool=shared_pool,
            )
        elif isinstance(effect, RaiseStandingBonusEffect):
            key = effect.scope_key
            if effect.scope_key_from_target:
                # "Give minions of **its type** in the Tavern +3/+3" — the
                # scope is read off the body the seat named, and the shop
                # dispatcher is never told one. The spell resolver handles it;
                # a minion carrying it would need a target this path cannot
                # supply, so say so rather than raise the wrong scope.
                raise UnhandledShopEffect(
                    f"{type(effect).__name__} with scope_key_from_target needs a "
                    "chosen target — resolve it through the spell path"
                )
            if effect.scope_kind is ScopeKind.CARD and key is None:
                # "for each other <me>" — the card scopes the bonus to itself.
                key = source.card_id if source is not None else None
            if not (effect.scope_kind is ScopeKind.CARD and key is None):
                # Everything reaching the shop dispatcher is outside a fight, so
                # this is where the second price on the card applies.
                raise_standing_bonus(
                    player,
                    BonusScope(effect.scope_kind, key, effect.scope_max_tier),
                    effect.attack_outside_combat or effect.attack,
                    effect.health_outside_combat or effect.health,
                )
        elif isinstance(effect, DiscoverTribeEffect) and effect.magnetize_onto_target:
            # Applied by apply_targeted_on_place_battlecries, which is the only
            # caller that knows which friendly the seat named.
            return
        elif isinstance(effect, DiscoverTribeEffect):
            self.open_tribe_discover(
                player,
                effect,
                shop_excluded_race=shop_excluded_race,
                shared_pool=shared_pool,
            )
        elif isinstance(effect, SetArmorEffect):
            player.armor = int(effect.amount)
        elif isinstance(effect, SetStatsEffect):
            # "Set a minion's stats to 20/20" — ``source`` is the body named,
            # which for a spell is the minion it was cast at.
            if source is not None:
                set_minion_stats(source, effect.attack, effect.health)
        elif isinstance(effect, AddSharedTribeMinionEffect):
            # ``source`` is the minion the spell was cast at; its tribe is the
            # whole instruction, which is why no named tribe can stand in.
            if source is None or source.race is None:
                return
            for _ in range(max(1, effect.count)):
                add_random_minion_to_hand(
                    player,
                    source.race,
                    shop_excluded_race,
                    rng=self._rng,
                    patch=self._patch,
                    exclude_card_id=source.card_id if effect.exclude_target else None,
                )
        elif isinstance(effect, CopyTavernMinionEffect):
            # ``source`` is the minion the spell was cast at — the tavern body
            # being copied. It stays where it is; only a copy changes hands.
            if source is None or source.card_id == effect.exclude_card_id:
                return
            for _ in range(max(1, effect.count)):
                slot = first_free_hand_slot(player)
                if slot is None:
                    break
                player.hand[slot] = make_minion(source.card_id, patch=self._patch)
        elif isinstance(effect, DealHeroDamagePerTierEffect):
            # ``placed`` is the card the Discover handed over, which is where
            # the number comes from — the binding cannot know it.
            if placed is not None:
                self.damage_hero(player, int(placed.tier) * max(1, effect.per_tier))
        elif isinstance(effect, GainStatsFromTavernEffect):
            self.apply_gain_stats_from_tavern(player, source, effect)
        elif isinstance(effect, CopyTargetingSpellEffect):
            # Handled by fire_spell_cast_on, which is the only caller that
            # knows which spell was cast; reaching here means nobody named it.
            return
        elif isinstance(effect, AddCardToNextRefreshesEffect):
            # "add **two** Fodders to your next 3 Refreshes" on the Golden:
            # more per roll, over the same number of rolls.
            have = player.refresh_promises.get(effect.card_id, 0)
            player.refresh_promises[effect.card_id] = have + int(
                effect.refreshes
            ) * max(1, int(effect.count))
        elif isinstance(effect, AddRandomCardToHandEffect):
            # The named set may be minions (the Chromadrakes) or spells (the
            # Bounties), so the card is looked up in both catalogs rather than
            # assumed to be one.
            pool = [
                cid
                for cid in effect.card_ids
                if cid in self._patch.templates or cid in self._patch.tavern_spells
            ]
            for _ in range(max(1, int(effect.count))):
                slot = first_free_hand_slot(player) if pool else None
                if slot is None:
                    break
                pick = pool[int(self._rng.integers(0, len(pool)))]
                player.hand[slot] = (
                    make_minion(pick, patch=self._patch)
                    if pick in self._patch.templates
                    else self._patch.tavern_spells[pick]
                )
        elif isinstance(effect, BuffShopOnEveryRefreshEffect):
            player.refresh_buffs = player.refresh_buffs + (
                (int(effect.attack), int(effect.health)),
            )
        elif isinstance(effect, ConsumeTavernMinionEffect):
            # Only the shapes that name their own eaters resolve here. The
            # seat-picked one (Mind Muck) is a targeted battlecry and is applied
            # by that path; it lands here too and finds nobody to feed, which is
            # what keeps it from eating twice.
            from src.bg_recruitment.targeted_battlecry import consume_tavern_minion

            if effect.each:
                eaters = [
                    m
                    for m in player.board
                    if effect.filter_race is None
                    or minion_matches_tribe(m, effect.filter_race)
                ]
            elif effect.eater_is_source and source is not None:
                # "Choose a friendly **Demon**" — the seat's pick is still held
                # to what the card asks for. A spell reaches here with its
                # target as the source, and without this a Beast ate.
                eaters = (
                    [source]
                    if effect.filter_race is None
                    or minion_matches_tribe(source, effect.filter_race)
                    else []
                )
            else:
                eaters = []
            for eater in eaters:
                for _ in range(max(1, effect.count)):
                    consume_tavern_minion(
                        player,
                        eater,
                        rng=self._rng,
                        highest_health=effect.highest_health,
                        stat_multiplier=effect.stat_multiplier,
                        gain_keywords=effect.gain_keywords,
                        shared_pool=shared_pool,
                    )
        elif isinstance(effect, StatsFromNextBuyEffect):
            if source is not None:
                source.wants_next_buy_stats = max(1, int(effect.factor))
        elif isinstance(effect, BuffHandMinionsEffect):
            _buff_hand_minions(player, effect)
        elif isinstance(effect, GiveOwnStatsToHandEffect):
            held = [
                c for c in player.hand if isinstance(c, Minion) and not is_locked(c)
            ]
            if held and source is not None:
                factor = max(1, int(effect.factor))
                held[0].bonus_attack += source.raw_attack * factor
                held[0].bonus_health += source.max_health * factor
        elif isinstance(effect, GrantCombinedChooseOneEffect):
            grant_combined_choose_one(player, effect.count)
        elif isinstance(effect, RaiseGoldCapEffect):
            player.ruleset = dataclass_replace(
                player.ruleset, gold_cap=player.ruleset.gold_cap + int(effect.amount)
            )
        elif isinstance(effect, AddRandomGoldenMinionEffect):
            from src.bg_catalog.patch_catalog import golden_upgrade_card_id

            catalog = self._patch.patch_dir / "catalog.json"
            pool = [
                cid
                for cid in self._patch.pool_ids
                if self._patch.templates[cid].tier == effect.tier
                and golden_upgrade_card_id(cid, catalog) is not None
            ]
            for _ in range(max(1, int(effect.count))):
                slot = first_free_hand_slot(player) if pool else None
                if slot is None:
                    break
                pick = pool[int(self._rng.integers(0, len(pool)))]
                made = make_minion(pick, patch=self._patch)
                from src.bg_recruitment.targeted_battlecry import make_golden

                # Minted rather than upgraded: the body is new, so the lobby
                # lends all three copies, not the two an upgrade needs.
                if shared_pool is not None and not shared_pool.acquire_new(pick, 3):
                    break
                make_golden(made, patch=self._patch)
                player.hand[slot] = made
        elif isinstance(effect, AddRandomMinionOfCommonTribeEffect):
            counts: dict = {}
            for m in player.board:
                if m.race is not None:
                    counts[m.race] = counts.get(m.race, 0) + 1
            if counts:
                tribe = max(counts.items(), key=lambda kv: kv[1])[0]
                add_random_minion_to_hand(
                    player,
                    tribe,
                    shop_excluded_race,
                    rng=self._rng,
                    patch=self._patch,
                )
        elif isinstance(effect, DoubleNextMagnetizeEffect):
            if source is not None:
                source.magnet_doubles_next = max(2, int(effect.factor))
        elif isinstance(effect, BuffPerMagnetizationEffect):
            for minion in player.board:
                if minion.magnetized_count <= 0:
                    continue
                minion.bonus_attack += effect.attack * minion.magnetized_count
                minion.bonus_health += effect.health * minion.magnetized_count
        elif isinstance(effect, GiveLockboxEffect):
            give_lockbox(player, sooner=int(effect.sooner))
        elif isinstance(effect, AddTavernSpellToHandEffect):
            spell = self._patch.tavern_spells.get(effect.card_id)
            for _ in range(max(1, int(effect.count))):
                slot = first_free_hand_slot(player) if spell is not None else None
                if slot is None:
                    break
                player.hand[slot] = spell
        elif isinstance(effect, ReduceTavernSpellCostEffect):
            player.tavern_spell_cost_delta -= int(effect.amount)
        elif isinstance(effect, GainGoldNextTurnEffect):
            player.gold_next_turn += int(effect.amount)
        elif isinstance(effect, AddTokenToHandEffect):
            for _ in range(max(0, effect.count)):
                slot = first_free_hand_slot(player)
                if slot is None:
                    break
                player.hand[slot] = make_minion(effect.token_id, patch=self._patch)
        elif isinstance(effect, IncrementShopTribeBonusEffect):
            if effect.tribe == Race.ELEMENTAL:
                player.shop_elemental_bonus += effect.attack
                player.shop_elemental_bonus_health += effect.health
            buff_shop_minions_of_tribe(
                player, effect.tribe, attack=effect.attack, health=effect.health
            )
        elif isinstance(effect, IncreaseTribeGiftEffect):
            player.elemental_gift_attack += effect.attack
            player.elemental_gift_health += effect.health
        elif isinstance(effect, AddFromLastOpponentBoardEffect):
            if not player.last_opponent_board:
                return
            pick = player.last_opponent_board[
                int(self._rng.integers(0, len(player.last_opponent_board)))
            ]
            slot = first_free_hand_slot(player)
            if slot is None:
                return
            if effect.make_golden:
                from src.bg_recruitment.triples import make_forged_golden_minion

                player.hand[slot] = make_forged_golden_minion(
                    pick.card_id, patch=self._patch
                )
            else:
                player.hand[slot] = make_minion(pick.card_id, patch=self._patch)
        elif isinstance(effect, TransformIntoShopMinionEffect):
            try:
                idx = player.board.index(source)
            except ValueError:
                return
            from src.bg_recruitment.faceless import apply_transform_into_shop_minion

            slots = [i for i, m in enumerate(player.shop) if m is not None]
            if not slots:
                return
            pick = slots[int(self._rng.integers(0, len(slots)))]
            apply_transform_into_shop_minion(
                player, idx, pick, patch=self._patch, copy_golden=effect.copy_golden
            )
        elif isinstance(effect, CreateSpellcraftSpellEffect):
            # Playing the Naga hands you its spell straight away; the start of
            # every later turn hands you another (fire_on_turn_start).
            give_spellcraft_spell(player, effect)
        elif isinstance(effect, AddRandomMinionToHandEffect) and effect.same_card:
            # "Get a random Murloc **and a copy of it**" — one roll, handed
            # over twice, not two rolls.
            before = [c for c in player.hand]
            add_random_minion_to_hand(
                player,
                effect.tribe,
                shop_excluded_race,
                rng=self._rng,
                patch=self._patch,
                tier=effect.tier,
                keyword=effect.keyword,
            )
            rolled = next(
                (
                    new
                    for old, new in zip(before, player.hand)
                    if old is None and new is not None
                ),
                None,
            )
            if rolled is not None:
                for _ in range(max(1, effect.count) - 1):
                    slot = first_free_hand_slot(player)
                    if slot is None:
                        break
                    player.hand[slot] = make_minion(rolled.card_id, patch=self._patch)
        elif isinstance(effect, AddRandomMinionToHandEffect):
            for _ in range(max(1, effect.count)):
                add_random_minion_to_hand(
                    player,
                    effect.tribe,
                    shop_excluded_race,
                    rng=self._rng,
                    patch=self._patch,
                    tier=effect.tier,
                    keyword=effect.keyword,
                )
        elif isinstance(effect, GainBloodGemsEffect):
            give_blood_gems(
                player, effect.count, quilboar_keyword=effect.quilboar_keyword
            )
        elif isinstance(effect, PlayBloodGemsEffect):
            for target in blood_gem_targets(player, source, effect.target):
                play_blood_gem_on(
                    player, target, count=effect.count, patch=self._patch
                )
        elif isinstance(effect, IncreaseBloodGemBonusEffect):
            player.blood_gem_bonus_attack += effect.attack
            player.blood_gem_bonus_health += effect.health
        else:
            raise UnhandledShopEffect(
                f"{type(effect).__name__} reached the shop dispatcher with no "
                f"handler and is not listed in _HANDLED_ELSEWHERE. Either give "
                f"it a branch or say where it is handled instead."
            )
        settle_standing_bonuses(player)
        refresh_attack_thresholds(player.board)

    def refresh_with_fishbait(
        self,
        player: PlayerState,
        source: Minion,
        *,
        shop_excluded_race: Optional[Race] = None,
        shared_pool=None,
    ) -> None:
        """"Refresh the Tavern with a Fishbait. Your left-most Beast attacks it."

        A reroll first, so the bait lands on a fresh counter, and then the
        exchange the tavern already knows how to run. ``source`` is the card
        being sold: it is a Beast itself and has not left the board yet, so the
        swing has to skip it.
        """
        from src.bg_recruitment.fishbait import attack_fishbait, place_fishbait
        from src.bg_recruitment.shop import refresh_shop

        refresh_shop(
            player,
            shop_excluded_race,
            rng=self._rng,
            shared_pool=shared_pool,
            patch=self._patch,
        )
        slot = next(
            (i for i, card in enumerate(player.shop) if card is not None),
            None,
        )
        if slot is None:
            return
        place_fishbait(player, slot, golden=source.is_golden)
        attack_fishbait(
            player,
            slot,
            exclude=source,
            fire_rally=lambda attacker: fire_tavern_rally(
                player,
                attacker,
                lambda src, eff: self.apply_shop_effect(player, src, eff, placed=None),
            ),
        )

    @staticmethod
    def _has_battlecry(minion: Minion) -> bool:
        return any(ab.trigger == Trigger.ON_PLACE for ab in minion.abilities)

    @staticmethod
    def _has_deathrattle(minion: Minion) -> bool:
        return any(ab.trigger == Trigger.ON_DEATH for ab in minion.abilities)

    def apply_magnetize_token(
        self, player: PlayerState, recipient: Minion, effect: MagnetizeTokenEffect
    ) -> None:
        """Weld a token onto ``recipient``, made on the spot.

        Two cards reach it from different events — a Mech being played, and a
        spell being cast on one — and they name different recipients, which is
        the whole reason this is a method rather than a branch inside the
        placement loop. It goes through the one magnetize entry point either
        way, so it counts, doubles and echoes like any other Magnetization.
        """
        from src.bg_recruitment.place import magnetize

        level = improve_level(player, effect.improves) if effect.improves else 1
        for _ in range(level * max(1, effect.repeats)):
            token = make_minion(effect.token_id, patch=self._patch)
            if effect.attack or effect.health:
                token.base_attack = effect.attack
                token.base_health = effect.health
            magnetize(player, recipient, token, triggers=self)
        if effect.improves:
            bump_seat_counter(player, effect.improves)

    def apply_gain_stats_from_tavern(
        self,
        player: PlayerState,
        source: Optional[Minion],
        effect: GainStatsFromTavernEffect,
    ) -> None:
        """"Gain the stats of the highest-Health minion in the Tavern."

        A read and not a purchase: the minion stays on the counter, which is
        the whole difference between this and consuming one.
        """
        from src.bg_recruitment.shop_auras import shop_effective_stats

        if source is None:
            return
        offered = [m for m in player.shop if m is not None]
        if not offered:
            return
        if effect.highest_health:
            biggest = max(offered, key=lambda m: m.max_health)
        else:
            biggest = offered[int(self._rng.integers(0, len(offered)))]
        attack, health = shop_effective_stats(offered, biggest)
        factor = max(1, int(effect.factor))
        source.bonus_attack += attack * factor
        source.bonus_health += health * factor

    def fire_elemental_played(self, player: PlayerState) -> None:
        """Count an Elemental against the watchers that answer every Nth.

        The seat plays it, but the countdown is each watcher's own — printed on
        the card as "(3 left!)", so two Tempests count separately.
        """
        for watcher in list(player.board):
            for ab in watcher.abilities:
                eff = ab.effect
                if not isinstance(eff, ElementalsPlayedResponseEffect):
                    continue
                watcher.elementals_seen += 1
                times, watcher.elementals_seen = divmod(
                    watcher.elementals_seen, max(1, int(eff.threshold))
                )
                for _ in range(times):
                    if eff.effect is not None:
                        self.apply_shop_effect(player, watcher, eff.effect, placed=None)

    def retrigger_friendly_ability(
        self,
        player: PlayerState,
        target: Minion,
        effect: RetriggerFriendlyAbilityEffect,
        *,
        shop_excluded_race: Optional[Race] = None,
        shared_pool=None,
    ) -> None:
        """Fire a friendly's own ability again — a Battlecry, or a Rally.

        Both go through the door that ability normally comes in by, so a
        re-trigger counts, multiplies and raises exactly as the first one did.
        """
        from src.bg_recruitment.fishbait import fire_tavern_rally

        for _ in range(max(1, effect.repeats)):
            if effect.trigger is Trigger.ON_ATTACK:
                fire_tavern_rally(
                    player,
                    target,
                    lambda src, eff: self.apply_shop_effect(
                        player, src, eff, placed=None
                    ),
                )
            else:
                self.fire_on_place(
                    placed=target,
                    player=player,
                    shop_excluded_race=shop_excluded_race,
                    shared_pool=shared_pool,
                )

    def open_tribe_discover(
        self,
        player: PlayerState,
        effect: DiscoverTribeEffect,
        *,
        repeats: int = 1,
        magnetize_onto_board_idx: Optional[int] = None,
        shop_excluded_race: Optional[Race] = None,
        shared_pool=None,
    ) -> None:
        """Park a "Discover a <tribe>" modal in front of the seat.

        Two cards reach it by different roads — a battlecry that opens it
        directly, and Maw Caster, which opens it as the payout for a body it
        destroyed — so it is a method rather than a branch inside the battlecry
        loop, which is where it used to live and where the second card silently
        found nothing.
        """
        from src.bg_recruitment.discover import try_open_hand_discover_modal

        total = max(1, repeats) * max(1, effect.repeats)
        if magnetize_onto_board_idx is None:
            # A pick that lands in hand needs somewhere to land.
            total = min(total, sum(1 for slot in player.hand if slot is None))
        if total <= 0:
            return
        tribe = effect.tribe
        if effect.most_common_tribe:
            # "a minion of your most common type" — read off the board when the
            # modal opens, so it is whatever the seat is actually playing.
            tribe = _most_common_tribe(player)
            if tribe is None:
                return
        elif effect.tribe_from_hero_rotation:
            # "Discover a minion of a specific minion type. Swaps type each
            # turn." — the type is the seat's hero's, rotated at turn start.
            tribe = player.hero_rotating_tribe
            if tribe is None:
                return
        opts = roll_discover_tribe_triple(
            self._rng,
            player.tavern_tier,
            shop_excluded_race,
            tribe=tribe,
            shared_pool=shared_pool,
            patch=self._patch,
            require_deathrattle=effect.require_deathrattle,
            require_battlecry=effect.require_battlecry,
            require_choose_one=effect.require_choose_one,
            exact_tier=effect.exact_tier,
        )
        if opts is None:
            return
        try_open_hand_discover_modal(
            player,
            PendingChoiceKind.DISCOVER_TRIBE,
            opts,
            total - 1,
            tribe=effect.tribe,
            magnetize_onto_board_idx=magnetize_onto_board_idx,
            lock_turns=effect.lock_turns,
            dies_if_played_this_turn=effect.dies_if_played_this_turn,
            shared_pool=shared_pool,
        )

    def fire_tavern_deathrattle(self, victim: Minion, player: PlayerState) -> None:
        """A minion destroyed in the tavern leaves its deathrattle behind.

        The modern patch prices this explicitly — Plaguerunner pays double "if
        triggered outside combat" — so a destroy in the recruit phase is a
        death that fires, not the silent removal it used to be. Only the effects
        with a tavern meaning run: a deathrattle that summons has nowhere to
        summon to, which is a property of the effect and so is declared once in
        ``_COMBAT_ONLY_ON_DEATH`` rather than guessed at per card.
        """
        for ab in victim.abilities:
            if ab.trigger != Trigger.ON_DEATH:
                continue
            bump_seat_counter(player, DEATHRATTLES_TRIGGERED)
            if isinstance(ab.effect, _COMBAT_ONLY_ON_DEATH):
                continue
            self.apply_shop_effect(player, victim, ab.effect, placed=None)

    def fire_on_sell(
        self,
        sold: Minion,
        player: PlayerState,
        *,
        shop_excluded_race: Optional[Race] = None,
        shared_pool=None,
    ) -> None:
        for ab in sold.abilities:
            if ab.trigger != Trigger.ON_SELL:
                continue
            self.apply_shop_effect(
                player,
                sold,
                ab.effect,
                sold,
                shop_excluded_race=shop_excluded_race,
                shared_pool=shared_pool,
            )
        self._fire_friendly_sold(sold, player)

    def _fire_friendly_sold(self, sold: Minion, player: PlayerState) -> None:
        """Board watchers of a sale ("after you sell an Elemental, gain +2/+2").

        Same trigger as the sold card's own effects and told apart by type: this
        one belongs to a minion staying behind, and the sold card is what it is
        watching.
        """
        for watcher in player.board:
            if watcher is sold:
                continue
            for ab in watcher.abilities:
                if ab.trigger != Trigger.ON_SELL:
                    continue
                eff = ab.effect
                if not isinstance(eff, BuffSelfOnFriendlySoldEffect):
                    continue
                if eff.filter_race is not None and not minion_matches_tribe(
                    sold, eff.filter_race
                ):
                    continue
                watcher.bonus_attack += eff.attack
                watcher.bonus_health += eff.health
                if eff.effect is not None:
                    self.apply_shop_effect(player, watcher, eff.effect, placed=None)

    def fire_on_friendly_bought(self, bought: Minion, player: PlayerState) -> None:
        if bought.race == Race.PIRATE:
            player.pirates_bought_this_turn += 1
        for m in list(player.board):
            if m is bought:
                continue
            for ab in m.abilities:
                if ab.trigger != Trigger.ON_FRIENDLY_BOUGHT:
                    continue
                if ab.filter_race is not None and not self.minion_matches_tribe(
                    bought, ab.filter_race
                ):
                    continue
                self.apply_shop_effect(player, m, ab.effect, bought)

    @staticmethod
    def battlecry_multiplier(board: List[Minion]) -> int:
        return multiplier_for(board, MultiplierKind.BATTLECRY)

    @staticmethod
    def summon_multiplier(board: List[Minion]) -> int:
        """Khadgar reads "your cards that summon minions summon twice as many"
        -- no phase qualifier -- so a tavern battlecry summon doubles exactly
        like a deathrattle summon in combat."""
        return multiplier_for(board, MultiplierKind.SUMMON)

    def fire_on_buy(self, minion: Minion, player: PlayerState) -> None:
        for ab in minion.abilities:
            if ab.trigger == Trigger.ON_BUY and isinstance(ab.effect, BuffRandomFriendly):
                self.apply_buff_random(minion, ab.effect, player.board)

    def fire_on_place(
        self,
        placed: Minion,
        player: PlayerState,
        shop_excluded_race: Optional[Race],
        *,
        shared_pool=None,
    ) -> None:
        mult = self.battlecry_multiplier(player.board)
        for ab in placed.abilities:
            if ab.trigger != Trigger.ON_PLACE:
                continue
            if not ability_condition_met(ab, player, player.board, placed=placed):
                continue
            e = ab.effect
            if isinstance(e, DiscoverTribeEffect) and not e.magnetize_onto_target:
                self.open_tribe_discover(
                    player,
                    e,
                    repeats=mult,
                    shop_excluded_race=shop_excluded_race,
                    shared_pool=shared_pool,
                )
                return
            if isinstance(e, ChooseOneEffect):
                # Parks the two options and stops here; the pick applies them,
                # multiplied battlecries included (Brann re-opens the choice).
                open_choose_one(player, e, source=placed)
                return
            if isinstance(e, AdaptAllMurlocsEffect):
                total = mult * e.repeats
                opts = roll_adapt_triple(self._rng)
                player.pending_choice = PendingChoice(
                    PendingChoiceKind.ADAPT, opts, total - 1
                )
                return
            if isinstance(e, AdaptSelfRandomEffect):
                if e.count_from_unique_other_tribes:
                    n = count_unique_tribes(player.board, exclude=placed) * mult * e.repeats
                else:
                    n = mult * e.repeats
                for _ in range(n):
                    key = ADAPT_KEYS_ALL[int(self._rng.integers(0, len(ADAPT_KEYS_ALL)))]
                    apply_adapt_key_to_minion(placed, key)
                continue
            if isinstance(e, TransformIntoShopMinionEffect):
                from src.bg_recruitment.faceless import try_open_transform_shop_modal

                idx = player.board.index(placed)
                try_open_transform_shop_modal(
                    player,
                    idx,
                    patch=self._patch,
                    rng=self._rng,
                    copy_golden=e.copy_golden,
                )
                return
        for ab in placed.abilities:
            if ab.trigger != Trigger.ON_PLACE:
                continue
            if isinstance(ab.effect, PogoHopperBattlecry):
                n = player.pogo_hoppers_played
                e = ab.effect
                placed.bonus_attack += e.attack_each * n * mult
                placed.bonus_health += e.health_each * n * mult
                player.pogo_hoppers_played += 1
                break
        i = 0
        while i < mult:
            i += 1
            for ab in placed.abilities:
                if ab.trigger != Trigger.ON_PLACE:
                    continue
                if not ability_condition_met(ab, player, player.board, placed=placed):
                    continue
                if isinstance(ab.effect, PogoHopperBattlecry):
                    continue
                if isinstance(ab.effect, AdaptSelfRandomEffect):
                    continue
                if isinstance(ab.effect, TransformIntoShopMinionEffect):
                    continue
                self.apply_shop_effect(
                    player, placed, ab.effect, placed,
                    shop_excluded_race=shop_excluded_race,
                    shared_pool=shared_pool,
                )
        # Spellcraft is a keyword, not a Battlecry: playing the Naga hands its
        # spell over straight away, and the start of every later turn hands
        # over another. Fired here rather than bound as an ON_PLACE ability
        # because Brann doubles those and does not double this.
        for ab in placed.abilities:
            if ab.trigger is Trigger.ON_TURN_START and isinstance(
                ab.effect, CreateSpellcraftSpellEffect
            ):
                give_spellcraft_spell(player, ab.effect)

        if minion_matches_tribe(placed, Race.ELEMENTAL):
            player.elementals_played += 1
            # The same count as a tally, for the cards that read one. The field
            # above stays because the observation has always read it.
            bump_seat_counter(player, "elementals_played:*")
            # ...and as a countdown each watcher keeps for itself.
            self.fire_elemental_played(player)
            from src.bg_recruitment import hero_passives

            hero_passives.apply_hero_on_elemental_played(player)  # Chenvaala

    def fire_after_friendly_minion_placed(
        self, player: PlayerState, placed: Minion
    ) -> None:
        self._fire_hand_watchers(player, placed)
        for m in list(player.board):
            for ab in m.abilities:
                if ab.trigger != Trigger.AFTER_FRIENDLY_MINION_PLACED:
                    continue
                # minion_matches_tribe, not ==: an Amalgam is every tribe, and
                # a bare comparison is how six of them stopped counting as any.
                if ab.filter_race is not None and not minion_matches_tribe(
                    placed, ab.filter_race
                ):
                    continue
                if ab.filter_max_tier and placed.tier > ab.filter_max_tier:
                    continue
                eff = ab.effect
                if isinstance(eff, MagnetizeTokenEffect):
                    self.apply_magnetize_token(player, placed, eff)
                    continue
                if isinstance(eff, BuffPlacedMinionEffect):
                    # The one listener on this trigger that pays the newcomer
                    # rather than the watcher.
                    if m is placed:
                        continue
                    placed.bonus_attack += eff.attack
                    placed.bonus_health += eff.health
                    continue
                if isinstance(eff, BuffSelfWhenFriendlyBattlecryPlaced):
                    if m is placed:
                        continue
                    if not self._has_battlecry(placed):
                        continue
                    m.bonus_attack += eff.attack
                    m.bonus_health += eff.health
                    continue
                if isinstance(eff, BuffSelfWhenFriendlyDeathrattlePlaced):
                    if m is placed:
                        continue
                    if not self._has_deathrattle(placed):
                        continue
                    m.bonus_attack += eff.attack
                    m.bonus_health += eff.health
                    continue
                if isinstance(eff, BuffSelfPerCount):
                    apply_buff_self_per_count(eff, m, player.board)
                    continue
                if isinstance(eff, BuffRandomFriendlyFromPlacedTierEffect):
                    e = eff
                    tier = max(0, placed.tier)
                    atk = tier if e.attack_per_tier else 0
                    hp = tier if e.health_per_tier else 0
                    if atk == 0 and hp == 0:
                        continue
                    eligible = [
                        x
                        for x in player.board
                        if not (e.exclude_self and x is m)
                    ]
                    if not eligible:
                        continue
                    pick = eligible[int(self._rng.integers(0, len(eligible)))]
                    pick.bonus_attack += atk
                    pick.bonus_health += hp
                    continue
                # The card says whether it wants a Battlecry, rather than the
                # target variant implying it for everyone.
                if getattr(eff, "requires_placed_battlecry", False):
                    if not self._has_battlecry(placed):
                        continue
                self.apply_shop_effect(player, m, ab.effect, placed)

    def fire_on_turn_end(self, player: PlayerState) -> None:
        # A Spellcraft spell is worth exactly the turn it was made for.
        discard_spellcraft_spells(player)
        # "Your end of turn effects trigger twice" (Drakkari Enchanter) — read
        # once, before anything fires, so a card that leaves the board partway
        # through does not change how often the rest of the pass runs.
        times = multiplier_for(player.board, MultiplierKind.END_OF_TURN)
        for _ in range(max(1, times)):
            self._fire_on_turn_end_pass(player)

    def _fire_on_turn_end_pass(self, player: PlayerState) -> None:
        for source in list(player.board):
            for ab in source.abilities:
                if ab.trigger != Trigger.ON_TURN_END:
                    continue
                if isinstance(ab.effect, BuffRandomFriendly):
                    self.apply_buff_random(source, ab.effect, player.board)
                elif isinstance(ab.effect, BuffOnePerListedTribeFriendly):
                    self.apply_buff_one_per_listed_tribe(
                        source, ab.effect, player.board
                    )
                elif isinstance(ab.effect, BuffSelf):
                    # "At the end of your turn, gain +1 Health" (Lullabot, and
                    # every card printed on that pattern since). fire_on_turn_start
                    # already delegates its BuffSelf the same way.
                    self.apply_shop_effect(player, source, ab.effect, None)
                elif isinstance(ab.effect, BuffSelfPerCount):
                    apply_buff_self_per_count(ab.effect, source, player.board)
                elif isinstance(ab.effect, BuffLeftmostRepeatedEffect):
                    e = ab.effect
                    n = int(getattr(player, e.counter, 0))
                    if n > 0 and player.board:
                        left = player.board[0]
                        for _ in range(n):
                            left.bonus_attack += e.attack
                            left.bonus_health += e.health
                else:
                    # Anything else this trigger can carry goes to the shop
                    # dispatcher, which raises on an effect nobody handles.
                    # The branches above stay because they read the board in
                    # ways the dispatcher's single-source signature cannot.
                    self.apply_shop_effect(player, source, ab.effect, None)

    @staticmethod
    def _find_by_promise_tag(player: PlayerState, tag: int):
        """The body a promise named, wherever the seat has put it since."""
        for card in list(player.board) + list(player.hand):
            if isinstance(card, Minion) and card.promise_tag == tag:
                return card
        return None

    def _pay_next_turn_promises(self, player: PlayerState) -> None:
        """Resolve what the seat was promised a turn ago, and let it go.

        The condition is asked here rather than when the promise was made, so
        "if you win your next combat" reads the fight that has since happened.
        A promise that named a body the seat no longer owns pays nothing.
        """
        owed, player.next_turn_promises = player.next_turn_promises, ()
        for promise, tag in owed:
            if promise.condition is not None and not condition_met(
                promise.condition, player, player.board
            ):
                continue
            named = self._find_by_promise_tag(player, tag) if tag else None
            if tag and named is None:
                continue
            from src.bg_recruitment.tavern_spells import apply_tavern_spell_effect

            for _ in range(max(1, int(promise.repeats))):
                apply_tavern_spell_effect(
                    player,
                    promise.effect,
                    rng=self._rng,
                    patch=self._patch,
                    source=named,
                )

    def fire_on_turn_start(self, player: PlayerState) -> None:
        """After round increment, before shop reroll: board L→R, then hand slots."""
        # The fight the promise was bought for has been and gone.
        player.start_combat_promises = ()
        # ...and a card discovered cheaply stops owing for it, played or not.
        for card in list(player.board) + list(player.hand):
            if isinstance(card, Minion) and card.dies_if_played_this_turn:
                card.dies_if_played_this_turn = False
        # ...and the other kind of promise comes due now, before anything else
        # this turn: "at the start of your next turn" is where the card puts it.
        self._pay_next_turn_promises(player)
        player.pirates_bought_this_turn = 0
        player.elementals_played = 0
        reset_activations(player)
        reset_health_refreshes(player)
        for minion in player.board:
            minion.magnet_doubles_next = 0
            minion.buy_answered_this_turn = False
        player.gold_spent_this_turn = 0
        # Last turn's "until next turn" buffs end here — after the combat they
        # were cast for, before the seat acts again.
        expire_temporary_buffs(player)
        # A Lockbox counts down in the seat's own turns. It only draws from the
        # generator on the turn it opens, and no shipped package can make one,
        # so the random stream on 36393 / 74257 is untouched.
        tick_lockboxes(player, rng=self._rng, patch=self._patch)
        # A held card counts down in the seat's own turns, the same way, so
        # "lock it in your hand for 1 turn" means one turn of yours.
        for card in player.hand:
            if is_locked(card):
                card.locked_turns -= 1
        for source in list(player.board):
            for ab in source.abilities:
                if ab.trigger != Trigger.ON_TURN_START:
                    continue
                if not ability_condition_met(ab, player, player.board, placed=source):
                    continue
                e = ab.effect
                if isinstance(e, BuffRandomFriendly):
                    self.apply_buff_random(source, e, player.board)
                elif isinstance(e, BuffOnePerListedTribeFriendly):
                    self.apply_buff_one_per_listed_tribe(source, e, player.board)
                elif isinstance(e, BuffSelf):
                    self.apply_shop_effect(player, source, e, None)
                elif isinstance(e, CreateSpellcraftSpellEffect):
                    give_spellcraft_spell(player, e)
                else:
                    # Same fallthrough fire_on_turn_end grew: anything else this
                    # trigger can carry goes to the dispatcher, which raises on
                    # an effect nobody handles rather than dropping it.
                    self.apply_shop_effect(player, source, e, None)
        for source in list(player.hand):
            if source is None:
                continue
            for ab in source.abilities:
                if ab.trigger != Trigger.ON_TURN_START:
                    continue
                e = ab.effect
                if isinstance(e, BuffRandomFriendly):
                    self.apply_buff_random(source, e, player.board)
                elif isinstance(e, BuffOnePerListedTribeFriendly):
                    self.apply_buff_one_per_listed_tribe(source, e, player.board)
                elif isinstance(e, BuffSelf):
                    self.apply_shop_effect(player, source, e, None)
